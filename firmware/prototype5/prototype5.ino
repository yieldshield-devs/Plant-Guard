/*
  ============================================================
                 SMART FARMING AI
                 ESP32-S3 MAIN NODE
  ============================================================

  ESP32-S3 CAM
       |
       | WiFi JPEG
       v
  ESP32-S3 MAIN
       |
       +---- Eggplant 7-class TFLite
       |
       +---- CAMERA NODE(S)
                    |
                    +---- OV3660 image
                    +---- Moisture sensor data
                    +---- WiFi -> MAIN NODE
       |
       +---- NPK RS485
       |
       +---- pH sensor
       |
       +---- DHT22
       |
       +---- Ultrasonic water level
       |
       v
     A7670C
       |
       +---- SMS -> 9830594081
       |
       +---- SMS -> 7003494215


  ============================================================
  PINOUT
  ============================================================

  A7670C
  --------
  GPIO17  <- A7670C TX
  GPIO18  -> A7670C RX
  GPIO16  -> A7670C PWRKEY


  NPK RS485
  ---------
  GPIO8   -> DI
  GPIO12  -> DE
  GPIO11  -> RE
  GPIO3   <- RO


  DHT22
  -----
  GPIO4   <- DATA


  pH SENSOR
  ---------
  GPIO2   <- PO / analog output
  GPIO42  <- T1


  ULTRASONIC
  ----------
  GPIO48  -> TRIG
  GPIO47  <- ECHO


  ============================================================
*/

#include <Arduino.h>
#include <WiFi.h>
#include <WebServer.h>
#include <JPEGDEC.h>
#include <DHT.h>
#include <esp_heap_caps.h>
#include <math.h>
#include <SPI.h>
#include <SD.h>
#include <ArduinoJson.h>

#include <Chirale_TensorFlowLite.h>
#include "tensorflow/lite/micro/all_ops_resolver.h"
#include "tensorflow/lite/micro/micro_interpreter.h"
#include "tensorflow/lite/schema/schema_generated.h"
#include "tensorflow/lite/c/common.h"


// ============================================================
// EGGPLANT MODEL
// ============================================================

extern const unsigned char g_model[];
extern const unsigned int g_model_len;


// ============================================================
// WIFI
// ============================================================

const char* SSID = "SmartFarm-S3";
const char* PASS = "12345678";

WebServer server(80);


// ============================================================
// MOISTURE DATA RECEIVED FROM CAMERA NODE(S)
// ============================================================

#define MAX_MOISTURE_NODES 10

float nodeMoisture[MAX_MOISTURE_NODES];
bool nodeMoistureValid[MAX_MOISTURE_NODES];
unsigned long nodeMoistureLastUpdate[MAX_MOISTURE_NODES];



// ============================================================
// A7670C
// ============================================================

HardwareSerial modem(1);

#define MODEM_RX       17
#define MODEM_TX       18
#define MODEM_PWRKEY   16

const char* PHONE_1 = "+919830594081";
const char* PHONE_2 = "+917003494215";

bool modemReady = false;

unsigned long imageCounter = 0;


// ============================================================
// NPK RS485
// ============================================================

HardwareSerial npkSerial(2);

#define NPK_DI  8
#define NPK_DE  12
#define NPK_RE  11
#define NPK_RO  3

#define NPK_BAUD 9600
#define NPK_SLAVE_ID 0x01

#define SOIL_MOISTURE_ALERT_PERCENT 60.0f


// ============================================================
// DHT22
// ============================================================

#define DHT_PIN 4
#define DHT_TYPE DHT22

DHT dht(DHT_PIN, DHT_TYPE);


// ============================================================
// pH SENSOR
// ============================================================

#define PH_PO_PIN 2
#define PH_T1_PIN 42

/*
  IMPORTANT:

  These are calibration values.

  PH7_VOLTAGE:
      Voltage measured from your sensor in pH 7 buffer.

  PH_SLOPE:
      Must be calibrated using known pH buffers.

  These are NOT universal values.
*/

float PH7_VOLTAGE = 2.50f;
float PH_SLOPE = 3.00f;


// ============================================================
// ULTRASONIC WATER LEVEL
// ============================================================

#define WATER_TRIG 48
#define WATER_ECHO 47

/*
  Distance from ultrasonic sensor to the
  bottom of the area being monitored.

  CHANGE THIS VALUE according to your setup.

  Example:
  Sensor is 12 inches above the bottom:
      12.0
*/

float WATER_EMPTY_DISTANCE_INCH = 12.0f;


// ============================================================
// TFLITE
// ============================================================

#define ARENA_SIZE (7 * 1024 * 1024)

uint8_t* arena = nullptr;

const tflite::Model* model = nullptr;

tflite::MicroInterpreter* interpreter = nullptr;

TfLiteTensor* input = nullptr;
TfLiteTensor* output = nullptr;

tflite::AllOpsResolver resolver;


// ============================================================
// EGGPLANT 7-CLASS MODEL
// ============================================================

#define NUM_CLASSES 7

const char* CLASS_NAMES[NUM_CLASSES] =
{
  "Healthy Leaf",
  "Insect Pest",
  "Leaf Spot",
  "Mosaic Virus",
  "Small Leaf",
  "White Mold",
  "Wilt Disease"
};

// JSON keys in /plant_guard_bangla_sms.json

const char* ADVICE_KEYS[NUM_CLASSES] =
{
  "",                  // Healthy Leaf -> no advice
  "পোকামাকড়",
  "পাতায় দাগ",
  "মোজাইক ভাইরাস",
  "ছোট পাতা",
  "সাদা ছত্রাক",
  "ঢলে পড়া রোগ"
};


// ============================================================
// JPEG
// ============================================================

#define MAX_JPEG_SIZE (256 * 1024)

uint8_t* jpg = nullptr;
size_t jpgSize = 0;

JPEGDEC jpeg;


// ============================================================
// PLANT GUARD ADVICE JSON ON SD
// ============================================================

#define SD_CS    37
#define SD_SCK   38
#define SD_MOSI  39
#define SD_MISO  40

#define ADVICE_JSON_PATH "/plant_guard_bangla_sms.json"

bool adviceJSONReady = false;
StaticJsonDocument<4096> adviceDoc;


// ============================================================
// SENSOR DATA
// ============================================================

struct NPKData
{
  float nitrogen;
  float phosphorus;
  float potassium;

  bool valid;
};


struct SensorData
{
  NPKData npk;

  float ph;

  float temperature;
  float humidity;

  float waterLevel;

  bool phValid;
  bool dhtValid;
  bool waterValid;
};


SensorData sensors;


// ============================================================
// MODEM READ
// ============================================================

String readModem(unsigned long timeout)
{
  String response = "";

  unsigned long start = millis();

  while (millis() - start < timeout)
  {
    while (modem.available())
    {
      char c = modem.read();

      response += c;

      Serial.write(c);
    }

    delay(5);
  }

  return response;
}


// ============================================================
// AT COMMAND
// ============================================================

bool sendAT(
  const char* command,
  const char* expected,
  unsigned long timeout = 3000
)
{
  Serial.println();
  Serial.print(">> ");
  Serial.println(command);

  while (modem.available())
    modem.read();

  modem.println(command);

  String response =
    readModem(timeout);

  if (response.indexOf(expected) >= 0)
  {
    Serial.println("AT COMMAND OK");
    return true;
  }

  Serial.println("AT COMMAND FAILED");

  return false;
}


// ============================================================
// MODEM POWER KEY
// ============================================================

void powerOnModem()
{
  pinMode(
    MODEM_PWRKEY,
    OUTPUT
  );

  digitalWrite(
    MODEM_PWRKEY,
    HIGH
  );

  delay(100);

  digitalWrite(
    MODEM_PWRKEY,
    LOW
  );

  delay(1200);

  digitalWrite(
    MODEM_PWRKEY,
    HIGH
  );

  delay(5000);
}


// ============================================================
// NETWORK
// ============================================================

bool waitForNetwork(
  unsigned long timeout
)
{
  Serial.println();
  Serial.println(
    "Waiting for network registration..."
  );

  unsigned long start =
    millis();

  while (
    millis() - start <
    timeout
  )
  {
    while (modem.available())
      modem.read();

    modem.println(
      "AT+CEREG?"
    );

    String response =
      readModem(1500);

    /*
      1 = registered
      5 = registered while roaming
    */

    if (
      response.indexOf(
        "+CEREG: 0,1"
      ) >= 0 ||
      response.indexOf(
        "+CEREG: 0,5"
      ) >= 0
    )
    {
      Serial.println(
        "NETWORK REGISTERED!"
      );

      return true;
    }

    Serial.println(
      "Not registered..."
    );

    delay(1500);
  }

  Serial.println(
    "NETWORK REGISTRATION TIMEOUT"
  );

  return false;
}


// ============================================================
// MODEM INIT
// ============================================================

bool initModem()
{
  Serial.println();
  Serial.println(
    "======================================"
  );
  Serial.println(
    "       INITIALIZING A7670C"
  );
  Serial.println(
    "======================================"
  );


  modem.begin(
    115200,
    SERIAL_8N1,
    MODEM_RX,
    MODEM_TX
  );


  Serial.println(
    "Powering A7670C..."
  );

  powerOnModem();


  Serial.println(
    "Waiting for modem..."
  );

  delay(3000);


  // Clear startup messages

  while (modem.available())
    modem.read();


  // ----------------------------------------------------------
  // FIND MODEM
  // ----------------------------------------------------------

  bool found = false;

  for (
    int attempt = 1;
    attempt <= 10;
    attempt++
  )
  {
    Serial.print(
      "AT attempt "
    );

    Serial.println(
      attempt
    );

    modem.println("AT");

    String response =
      readModem(2000);

    if (
      response.indexOf("OK")
      >= 0
    )
    {
      found = true;

      Serial.println(
        "A7670C ONLINE!"
      );

      break;
    }

    delay(1000);
  }


  if (!found)
  {
    Serial.println(
      "A7670C NOT RESPONDING!"
    );

    modemReady = false;

    return false;
  }


  // ----------------------------------------------------------
  // ECHO OFF
  // ----------------------------------------------------------

  sendAT(
    "ATE0",
    "OK",
    3000
  );


  // ----------------------------------------------------------
  // SIM
  // ----------------------------------------------------------

  bool simReady = false;

  for (
    int attempt = 1;
    attempt <= 10;
    attempt++
  )
  {
    modem.println(
      "AT+CPIN?"
    );

    String response =
      readModem(2000);

    if (
      response.indexOf(
        "READY"
      ) >= 0
    )
    {
      simReady = true;

      Serial.println(
        "SIM READY!"
      );

      break;
    }

    Serial.println(
      "SIM not ready..."
    );

    delay(1000);
  }


  if (!simReady)
  {
    Serial.println(
      "SIM NOT READY!"
    );

    modemReady = false;

    return false;
  }


  // ----------------------------------------------------------
  // SIGNAL
  // ----------------------------------------------------------

  sendAT(
    "AT+CSQ",
    "OK",
    3000
  );


  // ----------------------------------------------------------
  // SMS MODE
  // ----------------------------------------------------------

  if (
    !sendAT(
      "AT+CMGF=1",
      "OK",
      3000
    )
  )
  {
    modemReady = false;

    return false;
  }


  sendAT(
    "AT+CSCS=\"UCS2\"",
    "OK",
    3000
  );


  modemReady = true;


  Serial.println();
  Serial.println(
    "A7670C READY!"
  );


  /*
    Network registration may take
    longer, so don't fail initialization
    just because registration is slow.
  */

  waitForNetwork(30000);


  return true;
}


// ============================================================
// RS485 TRANSMIT
// ============================================================

void rs485Transmit()
{
  digitalWrite(
    NPK_RE,
    HIGH
  );

  digitalWrite(
    NPK_DE,
    HIGH
  );
}


// ============================================================
// RS485 RECEIVE
// ============================================================

void rs485Receive()
{
  digitalWrite(
    NPK_DE,
    LOW
  );

  digitalWrite(
    NPK_RE,
    LOW
  );
}


// ============================================================
// MODBUS CRC
// ============================================================

uint16_t modbusCRC(
  uint8_t* buffer,
  uint8_t length
)
{
  uint16_t crc = 0xFFFF;

  for (
    uint8_t pos = 0;
    pos < length;
    pos++
  )
  {
    crc ^= buffer[pos];

    for (
      uint8_t i = 0;
      i < 8;
      i++
    )
    {
      if (
        crc & 0x0001
      )
      {
        crc =
          (crc >> 1) ^
          0xA001;
      }
      else
      {
        crc >>= 1;
      }
    }
  }

  return crc;
}


// ============================================================
// READ NPK
// ============================================================

NPKData readNPK()
{
  NPKData result;
  result.nitrogen = NAN;
  result.phosphorus = NAN;
  result.potassium = NAN;
  result.valid = false;

  // These are the exact Modbus commands used by the
  // known-working Arduino Nano NPK code.
  const uint8_t nitrogenCmd[8]   = {0x01, 0x03, 0x00, 0x1E, 0x00, 0x01, 0xE4, 0x0C};
  const uint8_t phosphorusCmd[8] = {0x01, 0x03, 0x00, 0x1F, 0x00, 0x01, 0xB5, 0xCC};
  const uint8_t potassiumCmd[8]  = {0x01, 0x03, 0x00, 0x20, 0x00, 0x01, 0x85, 0xC0};

  auto readOne = [&](const uint8_t *cmd, const char *name, float &value) -> bool
  {
    uint8_t response[7];

    while (npkSerial.available())
      npkSerial.read();

    // Transmit
    rs485Transmit();
    delay(10);
    npkSerial.write(cmd, 8);
    npkSerial.flush();

    // Receive
    rs485Receive();

    unsigned long startTime = millis();
    uint8_t index = 0;

    while (millis() - startTime < 1000 && index < 7)
    {
      if (npkSerial.available())
        response[index++] = npkSerial.read();
    }

    if (index < 7)
    {
      Serial.print("NPK ");
      Serial.print(name);
      Serial.println(": TIMEOUT");
      return false;
    }

    Serial.print("NPK ");
    Serial.print(name);
    Serial.print(" RAW: ");
    for (uint8_t i = 0; i < 7; i++)
    {
      if (response[i] < 0x10) Serial.print('0');
      Serial.print(response[i], HEX);
      Serial.print(' ');
    }
    Serial.println();

    // Expected response: 01 03 02 XX XX CRC CRC
    if (response[0] != NPK_SLAVE_ID || response[1] != 0x03 || response[2] != 0x02)
    {
      Serial.print("NPK ");
      Serial.print(name);
      Serial.println(": INVALID RESPONSE");
      return false;
    }

    uint16_t receivedCRC = response[5] | ((uint16_t)response[6] << 8);
    uint16_t calculatedCRC = modbusCRC(response, 5);

    if (receivedCRC != calculatedCRC)
    {
      Serial.print("NPK ");
      Serial.print(name);
      Serial.println(": CRC ERROR");
      return false;
    }

    value = (float)(((uint16_t)response[3] << 8) | response[4]);
    return true;
  };

  bool nOK = readOne(nitrogenCmd, "N", result.nitrogen);
  delay(250);
  bool pOK = readOne(phosphorusCmd, "P", result.phosphorus);
  delay(250);
  bool kOK = readOne(potassiumCmd, "K", result.potassium);

  result.valid = nOK && pOK && kOK;

  if (result.valid)
  {
    Serial.print("N = "); Serial.print(result.nitrogen, 0); Serial.println(" ppm");
    Serial.print("P = "); Serial.print(result.phosphorus, 0); Serial.println(" ppm");
    Serial.print("K = "); Serial.print(result.potassium, 0); Serial.println(" ppm");
  }

  return result;
}


// ============================================================
// DHT22
// ============================================================

void readDHT22()
{
  float humidity =
    dht.readHumidity();

  float temperature =
    dht.readTemperature();


  if (
    isnan(humidity) ||
    isnan(temperature)
  )
  {
    sensors.dhtValid = false;

    Serial.println(
      "DHT22 ERROR"
    );

    return;
  }


  sensors.temperature =
    temperature;

  sensors.humidity =
    humidity;

  sensors.dhtValid =
    true;


  Serial.print(
    "Temperature: "
  );

  Serial.print(
    temperature,
    1
  );

  Serial.println(
    " C"
  );


  Serial.print(
    "Humidity: "
  );

  Serial.print(
    humidity,
    1
  );

  Serial.println(
    " %"
  );
}


// ============================================================
// pH
// ============================================================

float readPHVoltage()
{
  const int samples = 30;

  uint32_t total = 0;


  for (
    int i = 0;
    i < samples;
    i++
  )
  {
    total +=
      analogRead(
        PH_PO_PIN
      );

    delay(3);
  }


  float average =
    total /
    (float)samples;


  float voltage =
    (
      average /
      4095.0f
    )
    * 3.3f;


  return voltage;
}


// ============================================================
// READ pH
// ============================================================

void readPH()
{
  float voltage =
    readPHVoltage();


  Serial.print(
    "pH voltage: "
  );

  Serial.print(
    voltage,
    3
  );

  Serial.println(
    " V"
  );


  /*
    Basic linear calibration.

    pH = 7 + (V7 - Vmeasured) * slope
  */

  float ph =
    7.0f +
    (
      PH7_VOLTAGE -
      voltage
    )
    * PH_SLOPE;


  // Limit to physical pH range

  if (ph < 0)
    ph = 0;

  if (ph > 14)
    ph = 14;


  sensors.ph =
    ph;

  sensors.phValid =
    true;


  Serial.print(
    "pH = "
  );

  Serial.println(
    ph,
    2
  );
}


// ============================================================
// ULTRASONIC DISTANCE
// ============================================================

float ultrasonicDistanceInch()
{
  digitalWrite(
    WATER_TRIG,
    LOW
  );

  delayMicroseconds(3);


  digitalWrite(
    WATER_TRIG,
    HIGH
  );

  delayMicroseconds(10);


  digitalWrite(
    WATER_TRIG,
    LOW
  );


  unsigned long duration =
    pulseIn(
      WATER_ECHO,
      HIGH,
      30000
    );


  if (
    duration == 0
  )
  {
    return NAN;
  }


  float distanceCM =
    duration *
    0.0343f /
    2.0f;


  float distanceINCH =
    distanceCM /
    2.54f;


  return distanceINCH;
}


// ============================================================
// WATER LEVEL
// ============================================================

void readWaterLevel()
{
  float distance =
    ultrasonicDistanceInch();


  if (
    isnan(distance)
  )
  {
    sensors.waterValid =
      false;

    Serial.println(
      "ULTRASONIC ERROR"
    );

    return;
  }


  /*
    Water level:

    sensor-to-bottom distance
    -
    sensor-to-water distance

    Example:

    Empty = 12 inch
    Current distance = 9 inch

    Water = 3 inch
  */

  float level =
    WATER_EMPTY_DISTANCE_INCH -
    distance;


  if (level < 0)
    level = 0;


  sensors.waterLevel =
    level;

  sensors.waterValid =
    true;


  Serial.print(
    "Water distance: "
  );

  Serial.print(
    distance,
    2
  );

  Serial.println(
    " inch"
  );


  Serial.print(
    "Stagnant water level: "
  );

  Serial.print(
    level,
    2
  );

  Serial.println(
    " inch"
  );
}


// ============================================================
// READ ALL SENSORS
// ============================================================

void readAllSensors()
{
  Serial.println();
  Serial.println(
    "======================================"
  );

  Serial.println(
    "       READING FARM SENSORS"
  );

  Serial.println(
    "======================================"
  );


  sensors.npk =
    readNPK();


  readPH();


  readDHT22();


  readWaterLevel();


  Serial.println(
    "======================================"
  );
}


// ============================================================
// CAMERA NODE MOISTURE RECEIVER
// ============================================================

// Camera node sends:
//   http://192.168.4.1/moisture?node=1&value=67
//
// Example:
//   Node 1 -> 67 %
//   Node 2 -> 54 %
//
// ============================================================

void handleMoisture()
{
  if (!server.hasArg("node") || !server.hasArg("value"))
  {
    server.send(
      400,
      "text/plain",
      "Missing node or value"
    );
    return;
  }

  int node = server.arg("node").toInt();
  float value = server.arg("value").toFloat();

  if (
    node < 1 ||
    node > MAX_MOISTURE_NODES ||
    value < 0.0f ||
    value > 100.0f
  )
  {
    server.send(
      400,
      "text/plain",
      "Invalid node or moisture value"
    );
    return;
  }

  int index = node - 1;

  nodeMoisture[index] = value;
  nodeMoistureValid[index] = true;
  nodeMoistureLastUpdate[index] = millis();


  Serial.println();
  Serial.println(
    "--------------------------------------"
  );
  Serial.print(
    "Camera Node "
  );
  Serial.print(node);
  Serial.print(
    ": "
  );
  Serial.print(
    value,
    1
  );
  Serial.println(
    " %"
  );
  Serial.println(
    "--------------------------------------"
  );

  server.send(
    200,
    "text/plain",
    "MOISTURE RECEIVED"
  );
}


// ============================================================
// LOAD BANGLA ADVICE JSON FROM SD
// ============================================================

bool loadAdviceJSON()
{
  Serial.println();
  Serial.println("Loading Plant Guard Bangla advice JSON...");

  SPI.begin(
    SD_SCK,
    SD_MISO,
    SD_MOSI,
    SD_CS
  );

  if (!SD.begin(SD_CS, SPI))
  {
    Serial.println("SD CARD INITIALIZATION FAILED!");
    adviceJSONReady = false;
    return false;
  }

  if (!SD.exists(ADVICE_JSON_PATH))
  {
    Serial.println("ADVICE JSON FILE NOT FOUND!");
    Serial.println("Copy plant_guard_bangla_sms.json to SD root.");
    adviceJSONReady = false;
    return false;
  }

  File file = SD.open(ADVICE_JSON_PATH, FILE_READ);

  if (!file)
  {
    Serial.println("FAILED TO OPEN ADVICE JSON!");
    adviceJSONReady = false;
    return false;
  }

  DeserializationError error =
    deserializeJson(adviceDoc, file);

  file.close();

  if (error)
  {
    Serial.print("JSON PARSE FAILED: ");
    Serial.println(error.c_str());
    adviceJSONReady = false;
    return false;
  }

  adviceJSONReady = true;

  Serial.println("PLANT GUARD ADVICE JSON READY!");
  return true;
}


// ============================================================
// GET ADVICE FOR DETECTED CLASS
// ============================================================

String getAdviceForClass(int classIndex)
{
  if (
    classIndex < 0 ||
    classIndex >= NUM_CLASSES
  )
  {
    return "";
  }

  // Healthy Leaf has no disease advice.
  if (classIndex == 0)
  {
    return "";
  }

  if (!adviceJSONReady)
  {
    return "পরামর্শের তথ্য পাওয়া যায়নি।";
  }

  const char* key = ADVICE_KEYS[classIndex];

  if (!key || key[0] == '\0')
  {
    return "";
  }

  const char* advice =
    adviceDoc[key];

  if (!advice)
  {
    Serial.print("Advice not found in JSON for: ");
    Serial.println(key);
    return "পরামর্শের তথ্য পাওয়া যায়নি।";
  }

  return String(advice);
}


// ============================================================
// UTF-8 -> UCS2 HEX FOR A7670C SMS
// ============================================================

String utf8ToUcs2Hex(const String& text)
{
  String hex = "";
  const uint8_t* data =
    reinterpret_cast<const uint8_t*>(text.c_str());

  size_t i = 0;
  size_t len = text.length();

  while (i < len)
  {
    uint32_t cp = 0;
    uint8_t c = data[i];

    if (c < 0x80)
    {
      cp = c;
      i++;
    }
    else if ((c & 0xE0) == 0xC0 && i + 1 < len)
    {
      cp =
        ((uint32_t)(c & 0x1F) << 6) |
        (uint32_t)(data[i + 1] & 0x3F);
      i += 2;
    }
    else if ((c & 0xF0) == 0xE0 && i + 2 < len)
    {
      cp =
        ((uint32_t)(c & 0x0F) << 12) |
        ((uint32_t)(data[i + 1] & 0x3F) << 6) |
        (uint32_t)(data[i + 2] & 0x3F);
      i += 3;
    }
    else if ((c & 0xF8) == 0xF0 && i + 3 < len)
    {
      cp =
        ((uint32_t)(c & 0x07) << 18) |
        ((uint32_t)(data[i + 1] & 0x3F) << 12) |
        ((uint32_t)(data[i + 2] & 0x3F) << 6) |
        (uint32_t)(data[i + 3] & 0x3F);
      i += 4;
    }
    else
    {
      // Skip malformed UTF-8 byte.
      i++;
      continue;
    }

    // Bengali advice uses BMP code points.
    if (cp <= 0xFFFF)
    {
      char buf[5];
      snprintf(
        buf,
        sizeof(buf),
        "%04lX",
        (unsigned long)cp
      );

      hex += buf;
    }
    else
    {
      // Convert non-BMP code point to UTF-16 surrogate pair.
      cp -= 0x10000;

      uint16_t high =
        0xD800 + (cp >> 10);

      uint16_t low =
        0xDC00 + (cp & 0x3FF);

      char buf[9];

      snprintf(
        buf,
        sizeof(buf),
        "%04X%04X",
        high,
        low
      );

      hex += buf;
    }
  }

  return hex;
}


// ============================================================
// BUILD SMS
// ============================================================

String buildSMS(String disease)
{
  String sms = "";

  // Find detected class for bilingual status/advice.
  int detectedClass = 0;
  for (int i = 0; i < NUM_CLASSES; i++)
  {
    if (disease == CLASS_NAMES[i])
    {
      detectedClass = i;
      break;
    }
  }

  const char* statusBangla[NUM_CLASSES] = {
    "সুস্থ পাতা",
    "পোকামাকড়",
    "পাতায় দাগ",
    "মোজাইক রোগ",
    "ছোট পাতা",
    "সাদা ছত্রাক",
    "ঢলে পড়া রোগ"
  };

  const char* adviceEnglish[NUM_CLASSES] = {
    "No disease management is required. Continue regular monitoring and proper field care.",
    "Remove heavily affected leaves/fruits and use pheromone traps and neem-based insect control where appropriate.",
    "Remove affected leaves and maintain field sanitation. Use a suitable fungicide if required.",
    "Remove the infected plant completely with its roots and take it away from the field. Burn it or bury it underground. Wash your hands thoroughly with soap before touching healthy plants.",
    "Remove severely affected plants. Control leafhoppers and keep the field free of weeds.",
    "Remove affected plant parts. Avoid waterlogging and use suitable biological control such as Trichoderma where appropriate.",
    "Remove affected plants. Maintain good drainage and use suitable biological control such as Trichoderma or Pseudomonas where appropriate."
  };

  const char* adviceBangla[NUM_CLASSES] = {
    "কোনও রোগ ব্যবস্থাপনার প্রয়োজন নেই। নিয়মিত পর্যবেক্ষণ ও সঠিক পরিচর্যা চালিয়ে যান।",
    "আক্রান্ত ডাল ও ফল তুলে নষ্ট করুন। ফেরোমোন ফাঁদ ও নিমজাতীয় কীটনাশক ব্যবহার করুন।",
    "আক্রান্ত পাতা তুলে নষ্ট করুন। জমি পরিষ্কার রাখুন এবং উপযুক্ত ছত্রাকনাশক ব্যবহার করুন।",
    "আক্রান্ত গাছটিকে শিকড়সহ সম্পূর্ণ উপড়ে ফেলুন এবং মাঠ থেকে দূরে নিয়ে গিয়ে পুড়িয়ে দিন বা মাটির নিচে পুঁতে দিন। সুস্থ গাছ স্পর্শ করার আগে হাত সাবান দিয়ে ভালো করে ধুয়ে নিন।",
    "আক্রান্ত গাছ দ্রুত তুলে নষ্ট করুন। লিফহপার নিয়ন্ত্রণ করুন এবং জমি আগাছামুক্ত রাখুন।",
    "আক্রান্ত অংশ তুলে নষ্ট করুন। জল জমতে দেবেন না এবং প্রয়োজনে ট্রাইকোডার্মা ব্যবহার করুন।",
    "আক্রান্ত গাছ তুলে নষ্ট করুন। জল নিষ্কাশন ভালো রাখুন এবং ট্রাইকোডার্মা/সিউডোমোনাস ব্যবহার করুন।"
  };

  sms += "PLANT GUARD ALERT\r\n";
  sms += "প্ল্যান্ট গার্ড সতর্কতা\r\n\r\n";

  // Image number
  if (imageCounter == 1) sms += "1st image received (প্রথম ছবি গ্রহণ করা হয়েছে)\r\n";
  else if (imageCounter == 2) sms += "2nd image received (দ্বিতীয় ছবি গ্রহণ করা হয়েছে)\r\n";
  else if (imageCounter == 3) sms += "3rd image received (তৃতীয় ছবি গ্রহণ করা হয়েছে)\r\n";
  else sms += String(imageCounter) + "th image received (" + String(imageCounter) + "তম ছবি গ্রহণ করা হয়েছে)\r\n";

  sms += "Plant Status: ";
  sms += CLASS_NAMES[detectedClass];
  sms += " (";
  sms += statusBangla[detectedClass];
  sms += ")\r\n";

  sms += "Suggestion: ";
  sms += adviceEnglish[detectedClass];
  sms += "\r\n";

  sms += "পরামর্শ: ";
  sms += adviceBangla[detectedClass];
  sms += "\r\n\r\n";

  // NPK
  if (sensors.npk.valid)
  {
    sms += "Nitrogen: ";
    sms += String(sensors.npk.nitrogen, 0);
    sms += " ppm (নাইট্রোজেন: ";
    sms += String(sensors.npk.nitrogen, 0);
    sms += " পিপিএম)\r\n";


    sms += "Phosphorus: ";
    sms += String(sensors.npk.phosphorus, 0);
    sms += " ppm (ফসফরাস: ";
    sms += String(sensors.npk.phosphorus, 0);
    sms += " পিপিএম)\r\n";


    sms += "Potassium: ";
    sms += String(sensors.npk.potassium, 0);
    sms += " ppm (পটাশিয়াম: ";
    sms += String(sensors.npk.potassium, 0);
    sms += " পিপিএম)\r\n";

  }
  else
  {
    sms += "Nitrogen: ERROR (নাইট্রোজেন: তথ্য পাওয়া যায়নি)\r\n";
    sms += "Phosphorus: ERROR (ফসফরাস: তথ্য পাওয়া যায়নি)\r\n";
    sms += "Potassium: ERROR (পটাশিয়াম: তথ্য পাওয়া যায়নি)\r\n";
  }

  // Water level
  sms += "Water Level: ";
  if (sensors.waterValid)
  {
    sms += String(sensors.waterLevel, 1);
    sms += " inch";
    sms += " (জলের স্তর: ";
    sms += String(sensors.waterLevel, 1);
    sms += " ইঞ্চি)\r\n";
  }
  else
  {
    sms += "Not available in this crop (জলের স্তর: এই ফসলে তথ্য পাওয়া যায়নি)\r\n";
  }

  // pH
  sms += "pH: ";
  if (sensors.phValid)
  {
    sms += String(sensors.ph, 1);
    sms += " (পিএইচ: ";
    sms += String(sensors.ph, 1);
    sms += ")\r\n";
  }
  else
  {
    sms += "ERROR (পিএইচ: তথ্য পাওয়া যায়নি)\r\n";
  }

  // Moisture: use Camera Node 1 moisture as the primary plant moisture value.
  float moisture = 0.0f;
  bool moistureValid = nodeMoistureValid[0];
  if (moistureValid) moisture = nodeMoisture[0];

  sms += "Moisture: ";
  sms += String(moisture, 1);
  sms += "% (মাটির আর্দ্রতা: ";
  sms += String(moisture, 1);
  sms += "%)\r\n";

  if (moistureValid && moisture < SOIL_MOISTURE_ALERT_PERCENT)
  {
    sms += "Water the Plant (গাছে জল দাও)\r\n";
  }

  return sms;
}


// ============================================================
// SEND SMS TO ONE NUMBER
// ============================================================

bool sendSMSNumber(
  const char* phoneNumber,
  const String& message
)
{
  Serial.println();
  Serial.println(
    "--------------------------------------"
  );

  Serial.print(
    "Sending SMS to: "
  );

  Serial.println(
    phoneNumber
  );


  // ----------------------------------------------------------
  // Check network
  // ----------------------------------------------------------

  if (
    !waitForNetwork(
      30000
    )
  )
  {
    Serial.println(
      "NETWORK NOT REGISTERED"
    );

    return false;
  }


  // ----------------------------------------------------------
  // Text mode
  // ----------------------------------------------------------

  if (
    !sendAT(
      "AT+CMGF=1",
      "OK",
      3000
    )
  )
  {
    return false;
  }


  // Clear modem

  while (
    modem.available()
  )
    modem.read();


  // ----------------------------------------------------------
  // CMGS - UCS2 MODE
  // ----------------------------------------------------------

  // Phone number is encoded as UCS2 because CSCS=UCS2.
  String phoneUCS2 =
    utf8ToUcs2Hex(
      String(phoneNumber)
    );

  modem.print(
    "AT+CMGS=\""
  );

  modem.print(
    phoneUCS2
  );

  modem.println(
    "\""
  );


  // ----------------------------------------------------------
  // Wait for >
  // ----------------------------------------------------------

  unsigned long start =
    millis();

  bool promptReceived =
    false;


  while (
    millis() - start <
    10000
  )
  {
    while (
      modem.available()
    )
    {
      char c =
        modem.read();

      Serial.write(c);


      if (
        c == '>'
      )
      {
        promptReceived =
          true;

        break;
      }
    }


    if (
      promptReceived
    )
      break;
  }


  if (
    !promptReceived
  )
  {
    Serial.println(
      "SMS PROMPT FAILED"
    );

    return false;
  }


  // ----------------------------------------------------------
  // Send message as UCS2 HEX
  // ----------------------------------------------------------

  String messageUCS2 =
    utf8ToUcs2Hex(message);

  modem.print(
    messageUCS2
  );


  // CTRL+Z

  modem.write(
    26
  );


  Serial.println();
  Serial.println(
    "Message submitted."
  );


  // ----------------------------------------------------------
  // Wait for confirmation
  // ----------------------------------------------------------

  String response = "";

  start =
    millis();


  while (
    millis() - start <
    30000
  )
  {
    while (
      modem.available()
    )
    {
      char c =
        modem.read();

      Serial.write(c);

      response += c;


      if (
        response.indexOf(
          "+CMGS:"
        ) >= 0 &&
        response.indexOf(
          "OK"
        ) >= 0
      )
      {
        Serial.println();
        Serial.println(
          "SMS SENT SUCCESSFULLY!"
        );

        return true;
      }


      if (
        response.indexOf(
          "ERROR"
        ) >= 0
      )
      {
        Serial.println();
        Serial.println(
          "SMS FAILED"
        );

        return false;
      }
    }
  }


  Serial.println(
    "SMS TIMEOUT"
  );

  return false;
}


// ============================================================
// SEND SMS TO BOTH NUMBERS
// ============================================================

bool sendSMS(
  String disease
)
{
  if (!modemReady)
  {
    Serial.println(
      "MODEM NOT READY"
    );

    return false;
  }


  // Read sensors immediately
  // before creating SMS.

  readAllSensors();


  // Create complete report

  String message =
    buildSMS(
      disease
    );


  Serial.println();
  Serial.println(
    "======================================"
  );

  Serial.println(
    "           SMS REPORT"
  );

  Serial.println(
    "======================================"
  );

  Serial.println(
    message
  );

  Serial.println(
    "======================================"
  );


  // ----------------------------------------------------------
  // NUMBER 1
  // ----------------------------------------------------------

  bool result1 =
    sendSMSNumber(
      PHONE_1,
      message
    );


  // Give modem a little breathing time

  delay(3000);


  // ----------------------------------------------------------
  // NUMBER 2
  // ----------------------------------------------------------

  bool result2 =
    sendSMSNumber(
      PHONE_2,
      message
    );


  Serial.println();
  Serial.println(
    "======================================"
  );


  if (
    result1 &&
    result2
  )
  {
    Serial.println(
      "SMS SENT TO BOTH NUMBERS"
    );
  }
  else if (
    result1
  )
  {
    Serial.println(
      "SMS SENT TO NUMBER 1 ONLY"
    );
  }
  else if (
    result2
  )
  {
    Serial.println(
      "SMS SENT TO NUMBER 2 ONLY"
    );
  }
  else
  {
    Serial.println(
      "SMS FAILED TO BOTH NUMBERS"
    );
  }


  Serial.println(
    "======================================"
  );


  return (
    result1 &&
    result2
  );
}


// ============================================================
// JPEG DRAW CALLBACK
// ============================================================

int drawJPEG(
  JPEGDRAW *pDraw
)
{
  if (!input)
    return 0;


  /*
    The original code was fixed to 224x224.

    We determine the model dimensions
    dynamically.
  */

  int inputHeight =
    input->dims->data[
      input->dims->size - 3
    ];

  int inputWidth =
    input->dims->data[
      input->dims->size - 2
    ];


  int sw =
    pDraw->iWidth;

  int sh =
    pDraw->iHeight;


  int cropSize =
    min(
      sw,
      sh
    );


  int cropX =
    (sw - cropSize) /
    2;


  int cropY =
    (sh - cropSize) /
    2;


  for (
    int y = 0;
    y < pDraw->iHeight;
    y++
  )
  {
    int sy =
      pDraw->y +
      y;


    if (
      sy < cropY ||
      sy >= cropY +
            cropSize
    )
      continue;


    int dy =
      (
        (sy - cropY) *
        inputHeight
      ) /
      cropSize;


    if (
      dy < 0 ||
      dy >= inputHeight
    )
      continue;


    for (
      int x = 0;
      x < pDraw->iWidth;
      x++
    )
    {
      int sx =
        pDraw->x +
        x;


      if (
        sx < cropX ||
        sx >= cropX +
              cropSize
      )
        continue;


      int dx =
        (
          (sx - cropX) *
          inputWidth
        ) /
        cropSize;


      if (
        dx < 0 ||
        dx >= inputWidth
      )
        continue;


      uint16_t c =
        pDraw->pPixels[
          y *
          pDraw->iWidth +
          x
        ];


      uint8_t r =
        (
          (c >> 11) &
          0x1F
        ) << 3;


      uint8_t g =
        (
          (c >> 5) &
          0x3F
        ) << 2;


      uint8_t b =
        (
          c &
          0x1F
        ) << 3;


      int index =
        (
          dy *
          inputWidth +
          dx
        ) * 3;


      // ------------------------------------------------------
      // UINT8 INPUT
      // ------------------------------------------------------

      if (
        input->type ==
        kTfLiteUInt8
      )
      {
        input->data.uint8[
          index
        ] = r;

        input->data.uint8[
          index + 1
        ] = g;

        input->data.uint8[
          index + 2
        ] = b;
      }


      // ------------------------------------------------------
      // INT8 INPUT
      // ------------------------------------------------------

      else if (
        input->type ==
        kTfLiteInt8
      )
      {
        float scale =
          input->params.scale;

        int zeroPoint =
          input->params.zero_point;


        int qr =
          roundf(
            r / scale
          ) +
          zeroPoint;


        int qg =
          roundf(
            g / scale
          ) +
          zeroPoint;


        int qb =
          roundf(
            b / scale
          ) +
          zeroPoint;


        qr =
          constrain(
            qr,
            -128,
            127
          );

        qg =
          constrain(
            qg,
            -128,
            127
          );

        qb =
          constrain(
            qb,
            -128,
            127
          );


        input->data.int8[
          index
        ] =
          (int8_t)qr;


        input->data.int8[
          index + 1
        ] =
          (int8_t)qg;


        input->data.int8[
          index + 2
        ] =
          (int8_t)qb;
      }
    }
  }


  return 1;
}


// ============================================================
// RUN ML
// ============================================================

String runML()
{
  Serial.println();
  Serial.println(
    "======================================"
  );

  Serial.println(
    "             RUNNING ML"
  );

  Serial.println(
    "======================================"
  );


  if (
    !interpreter ||
    !input ||
    !output
  )
  {
    Serial.println(
      "ML NOT INITIALIZED"
    );

    return "ML_ERROR";
  }


  if (
    jpgSize == 0
  )
  {
    return "ML_ERROR";
  }


  // Clear input

  memset(
    input->data.raw,
    0,
    input->bytes
  );


  // ----------------------------------------------------------
  // Open JPEG
  // ----------------------------------------------------------

  if (
    !jpeg.openRAM(
      jpg,
      jpgSize,
      drawJPEG
    )
  )
  {
    Serial.println(
      "JPEG OPEN FAILED"
    );

    return "ML_ERROR";
  }


  Serial.print(
    "JPEG: "
  );

  Serial.print(
    jpeg.getWidth()
  );

  Serial.print(
    " x "
  );

  Serial.println(
    jpeg.getHeight()
  );


  // ----------------------------------------------------------
  // Decode
  // ----------------------------------------------------------

  if (
    !jpeg.decode(
      0,
      0,
      0
    )
  )
  {
    Serial.println(
      "JPEG DECODE FAILED"
    );

    jpeg.close();

    return "ML_ERROR";
  }


  jpeg.close();


  Serial.println(
    "JPEG PREPROCESSING DONE"
  );


  // ----------------------------------------------------------
  // INPUT INFO
  // ----------------------------------------------------------

  Serial.print(
    "Input type: "
  );

  Serial.println(
    input->type
  );


  Serial.print(
    "Input bytes: "
  );

  Serial.println(
    input->bytes
  );


  Serial.print(
    "Input scale: "
  );

  Serial.println(
    input->params.scale,
    8
  );


  Serial.print(
    "Input zero point: "
  );

  Serial.println(
    input->params.zero_point
  );


  // ----------------------------------------------------------
  // INFERENCE
  // ----------------------------------------------------------

  unsigned long start =
    millis();


  if (
    interpreter->Invoke()
    !=
    kTfLiteOk
  )
  {
    Serial.println(
      "TFLITE INVOKE FAILED"
    );

    return "ML_ERROR";
  }


  unsigned long inferenceTime =
    millis() - start;


  Serial.print(
    "Inference time: "
  );

  Serial.print(
    inferenceTime
  );

  Serial.println(
    " ms"
  );


  // ----------------------------------------------------------
  // FIND BEST CLASS
  // ----------------------------------------------------------

  int bestClass =
    0;


  float bestScore =
    -999999.0f;


  Serial.println();
  Serial.println(
    "========== EGGPLANT RESULTS =========="
  );


  for (
    int i = 0;
    i < NUM_CLASSES;
    i++
  )
  {
    float score =
      0;


    if (
      output->type ==
      kTfLiteFloat32
    )
    {
      score =
        output->data.f[
          i
        ];
    }


    else if (
      output->type ==
      kTfLiteUInt8
    )
    {
      score =
        (
          output->data.uint8[
            i
          ]
          -
          output->params.zero_point
        )
        *
        output->params.scale;
    }


    else if (
      output->type ==
      kTfLiteInt8
    )
    {
      score =
        (
          output->data.int8[
            i
          ]
          -
          output->params.zero_point
        )
        *
        output->params.scale;
    }


    Serial.print(
      i
    );

    Serial.print(
      " : "
    );

    Serial.print(
      CLASS_NAMES[i]
    );

    Serial.print(
      " = "
    );

    Serial.println(
      score,
      6
    );


    if (
      score >
      bestScore
    )
    {
      bestScore =
        score;

      bestClass =
        i;
    }
  }


  Serial.println(
    "===================================="
  );


  Serial.print(
    "BEST CLASS: "
  );

  Serial.println(
    CLASS_NAMES[
      bestClass
    ]
  );


  Serial.print(
    "BEST SCORE: "
  );

  Serial.println(
    bestScore,
    6
  );


  // ----------------------------------------------------------
  // SEND SENSOR REPORT
  // ----------------------------------------------------------

  String result =
    CLASS_NAMES[
      bestClass
    ];


  sendSMS(
    result
  );


  return result;
}


// ============================================================
// WEB HOME
// ============================================================

void home()
{
  String page = "";

  page +=
    "<html>"
    "<head>"
    "<title>Smart Farm</title>"
    "</head>"
    "<body>";

  page +=
    "<h1>Smart Farming AI</h1>";

  page +=
    "<h2>Eggplant Disease Detection</h2>";

  page +=
    "<form method='POST' "
    "action='/upload' "
    "enctype='multipart/form-data'>";

  page +=
    "<input type='file' "
    "name='photo' "
    "accept='image/jpeg' "
    "required>";

  page +=
    "<br><br>";

  page +=
    "<input type='submit' "
    "value='Analyze'>";

  page +=
    "</form>";


  page +=
    "<h2>Sensor Data</h2>";


  if (
    sensors.npk.valid
  )
  {
    page +=
      "Nitrogen: " +
      String(
        sensors.npk.nitrogen,
        0
      ) +
      " ppm<br>";

    page +=
      "Phosphorus: " +
      String(
        sensors.npk.phosphorus,
        0
      ) +
      " ppm<br>";

    page +=
      "Potassium: " +
      String(
        sensors.npk.potassium,
        0
      ) +
      " ppm<br>";
  }


  page +=
    "pH: " +
    String(
      sensors.ph,
      2
    ) +
    "<br>";


  page +=
    "Temperature: " +
    String(
      sensors.temperature,
      1
    ) +
    " C<br>";


  page +=
    "Humidity: " +
    String(
      sensors.humidity,
      1
    ) +
    " %<br>";


  page +=
    "Stagnant Water: " +
    String(
      sensors.waterLevel,
      2
    ) +
    " inch<br>";


  page +=
    "</body>"
    "</html>";


  server.send(
    200,
    "text/html",
    page
  );
}


// ============================================================
// PHONE IMAGE UPLOAD
// ============================================================

void upload()
{
  HTTPUpload& u =
    server.upload();


  if (
    u.status ==
    UPLOAD_FILE_START
  )
  {
    jpgSize = 0;


    if (!jpg)
    {
      jpg =
        (uint8_t*)
        heap_caps_malloc(
          MAX_JPEG_SIZE,
          MALLOC_CAP_SPIRAM
        );
    }


    Serial.println();
    Serial.println(
      "NEW IMAGE"
    );


    Serial.print(
      "Filename: "
    );

    Serial.println(
      u.filename
    );
  }


  else if (
    u.status ==
    UPLOAD_FILE_WRITE
  )
  {
    if (
      jpgSize +
      u.currentSize
      <=
      MAX_JPEG_SIZE
    )
    {
      memcpy(
        jpg + jpgSize,
        u.buf,
        u.currentSize
      );


      jpgSize +=
        u.currentSize;
    }
  }


  else if (
    u.status ==
    UPLOAD_FILE_END
  )
  {
    Serial.print(
      "JPEG received: "
    );

    Serial.print(
      jpgSize
    );

    Serial.println(
      " bytes"
    );
  }
}


// ============================================================
// CAMERA RAW JPEG
// ============================================================

void cameraRaw()
{
  HTTPRaw& raw =
    server.raw();


  if (
    raw.status ==
    RAW_START
  )
  {
    jpgSize = 0;


    if (!jpg)
    {
      jpg =
        (uint8_t*)
        heap_caps_malloc(
          MAX_JPEG_SIZE,
          MALLOC_CAP_SPIRAM
        );
    }


    Serial.println();
    Serial.println(
      "CAMERA PHOTO START"
    );


    Serial.print(
      "Expected size: "
    );

    Serial.println(
      raw.totalSize
    );
  }


  else if (
    raw.status ==
    RAW_WRITE
  )
  {
    if (
      jpgSize +
      raw.currentSize
      <=
      MAX_JPEG_SIZE
    )
    {
      memcpy(
        jpg + jpgSize,
        raw.buf,
        raw.currentSize
      );


      jpgSize +=
        raw.currentSize;
    }
    else
    {
      Serial.println(
        "JPEG TOO LARGE"
      );
    }
  }


  else if (
    raw.status ==
    RAW_END
  )
  {
    Serial.print(
      "Camera JPEG received: "
    );

    Serial.print(
      jpgSize
    );

    Serial.println(
      " bytes"
    );
  }
}


// ============================================================
// CAMERA REQUEST COMPLETE
// ============================================================

void cameraFinished()
{
  if (
    jpgSize == 0
  )
  {
    server.send(
      400,
      "text/plain",
      "No JPEG received"
    );

    return;
  }


  Serial.println();
  Serial.println(
    "PROCESSING CAMERA PHOTO"
  );


  imageCounter++;

  String result =
    runML();


  if (
    result ==
    "ML_ERROR"
  )
  {
    server.send(
      500,
      "text/plain",
      "ML FAILED"
    );

    jpgSize = 0;

    return;
  }


  server.send(
    200,
    "text/plain",
    "CAMERA RESULT: " +
    result
  );


  jpgSize = 0;
}


// ============================================================
// SETUP
// ============================================================

void setup()
{
  Serial.begin(
    115200
  );


  delay(2000);


  // ==========================================================
  // INITIALIZE CAMERA NODE MOISTURE DATA
  // ==========================================================

  for (
    int i = 0;
    i < MAX_MOISTURE_NODES;
    i++
  )
  {
    nodeMoisture[i] = 0.0f;
    nodeMoistureValid[i] = false;
    nodeMoistureLastUpdate[i] = 0;
  }


  Serial.println();
  Serial.println();
  Serial.println(
    "======================================"
  );

  Serial.println(
    "       SMART FARMING AI"
  );

  Serial.println(
    "       ESP32-S3 MAIN NODE"
  );

  Serial.println(
    "======================================"
  );


  // ==========================================================
  // PSRAM
  // ==========================================================

  Serial.print(
    "PSRAM: "
  );


  if (
    psramFound()
  )
  {
    Serial.println(
      "YES"
    );
  }
  else
  {
    Serial.println(
      "NO"
    );


    Serial.println(
      "PSRAM REQUIRED!"
    );


    while (1)
      delay(1000);
  }


  Serial.print(
    "Total PSRAM: "
  );

  Serial.println(
    ESP.getPsramSize()
  );


  Serial.print(
    "Free PSRAM: "
  );

  Serial.println(
    ESP.getFreePsram()
  );


  // ==========================================================
  // NPK RS485
  // ==========================================================

  pinMode(
    NPK_DE,
    OUTPUT
  );

  pinMode(
    NPK_RE,
    OUTPUT
  );


  rs485Receive();


  npkSerial.begin(
    NPK_BAUD,
    SERIAL_8N1,
    NPK_RO,
    NPK_DI
  );


  Serial.println(
    "NPK RS485 READY"
  );


  // ==========================================================
  // DHT22
  // ==========================================================

  dht.begin();


  Serial.println(
    "DHT22 READY"
  );


  // ==========================================================
  // pH
  // ==========================================================

  analogReadResolution(
    12
  );


  pinMode(
    PH_PO_PIN,
    INPUT
  );


  pinMode(
    PH_T1_PIN,
    INPUT
  );


  Serial.println(
    "pH SENSOR READY"
  );


  // ==========================================================
  // ULTRASONIC
  // ==========================================================

  pinMode(
    WATER_TRIG,
    OUTPUT
  );

  pinMode(
    WATER_ECHO,
    INPUT
  );


  digitalWrite(
    WATER_TRIG,
    LOW
  );


  Serial.println(
    "ULTRASONIC READY"
  );


  // ==========================================================
  // JPEG BUFFER
  // ==========================================================

  jpg =
    (uint8_t*)
    heap_caps_malloc(
      MAX_JPEG_SIZE,
      MALLOC_CAP_SPIRAM
    );


  if (!jpg)
  {
    Serial.println(
      "JPEG BUFFER FAILED!"
    );
  }
  else
  {
    Serial.println(
      "JPEG BUFFER READY"
    );
  }


  // ==========================================================
  // TFLITE MODEL
  // ==========================================================

  Serial.println();
  Serial.println(
    "Loading Eggplant 7-class model..."
  );


  Serial.print(
    "Model size: "
  );

  Serial.println(
    g_model_len
  );


  model =
    tflite::GetModel(
      g_model
    );


  if (
    model->version() !=
    TFLITE_SCHEMA_VERSION
  )
  {
    Serial.println(
      "MODEL SCHEMA ERROR!"
    );

    while (1)
      delay(1000);
  }


  arena =
    (uint8_t*)
    heap_caps_malloc(
      ARENA_SIZE,
      MALLOC_CAP_SPIRAM
    );


  if (!arena)
  {
    Serial.println(
      "TENSOR ARENA FAILED!"
    );

    while (1)
      delay(1000);
  }


  static tflite::MicroInterpreter staticInterpreter(
    model,
    resolver,
    arena,
    ARENA_SIZE
  );


  interpreter =
    &staticInterpreter;


  if (
    interpreter->AllocateTensors()
    !=
    kTfLiteOk
  )
  {
    Serial.println(
      "ALLOCATE TENSORS FAILED!"
    );

    while (1)
      delay(1000);
  }


  input =
    interpreter->input(0);


  output =
    interpreter->output(0);


  Serial.println(
    "EGGPLANT 7-CLASS MODEL READY!"
  );


  Serial.print(
    "Input type: "
  );

  Serial.println(
    input->type
  );


  Serial.print(
    "Input bytes: "
  );

  Serial.println(
    input->bytes
  );


  Serial.print(
    "Output type: "
  );

  Serial.println(
    output->type
  );


  // ==========================================================
  // PLANT GUARD ADVICE JSON
  // ==========================================================

  loadAdviceJSON();


  // ==========================================================
  // A7670C
  // ==========================================================

  initModem();


  // ==========================================================
  // WIFI AP
  // ==========================================================

  WiFi.mode(
    WIFI_AP
  );


  WiFi.softAP(
    SSID,
    PASS
  );


  Serial.print(
    "WiFi AP IP: "
  );

  Serial.println(
    WiFi.softAPIP()
  );


  // ==========================================================
  // WEB SERVER
  // ==========================================================

  server.on(
    "/",
    HTTP_GET,
    home
  );


  // CAMERA NODE SENDS MOISTURE HERE, THEN SENDS ITS JPEG TO /camera.
  // Both values are stored on the God Node and combined into the same report.
  server.on(
    "/moisture",
    HTTP_GET,
    handleMoisture
  );


  server.on(
    "/upload",
    HTTP_POST,

    []()
    {
      String result =
        runML();

      server.send(
        200,
        "text/plain",
        "RESULT: " +
        result
      );

      jpgSize = 0;
    },

    upload
  );


  /*
    CAMERA NODE sends its moisture to /moisture first, then POSTs
    its raw OV3660 JPEG to /camera.

       /moisture?node=1&value=67
       /camera   (raw JPEG)

    The God Node keeps the latest camera-node moisture value and
    includes it with the ML result in the same Plant Guard SMS.
  */

  server.on(
    "/camera",
    HTTP_POST,

    cameraFinished,

    cameraRaw
  );


  server.begin();


  Serial.println(
    "WEB SERVER READY"
  );


  // ==========================================================
  // INITIAL SENSOR READ
  // ==========================================================

  readAllSensors();


  Serial.println();
  Serial.println(
    "======================================"
  );

  Serial.println(
    "          SYSTEM READY"
  );

  Serial.println(
    "======================================"
  );
}


// ============================================================
// LOOP
// ============================================================

unsigned long lastSensorRead =
  0;


void loop()
{
  server.handleClient();


  // ----------------------------------------------------------
  // Update sensors every 10 seconds
  // ----------------------------------------------------------

  if (
    millis() -
    lastSensorRead >
    10000
  )
  {
    lastSensorRead =
      millis();


    readAllSensors();
  }


  // ----------------------------------------------------------
  // Manual modem AT commands
  // ----------------------------------------------------------

  if (
    Serial.available()
  )
  {
    String command =
      Serial.readStringUntil(
        '\n'
      );


    command.trim();


    if (
      command.length() > 0
    )
    {
      modem.println(
        command
      );
    }
  }
}