#include "esp_camera.h"
#include <WiFi.h>
#include <HTTPClient.h>

const char* SSID = "SmartFarm-S3";
const char* PASSWORD = "12345678";

const char* CAMERA_SERVER = "http://192.168.4.1/camera";
const char* MOISTURE_SERVER = "http://192.168.4.1/moisture";

#define NODE_ID 1

// CHANGE THIS TO YOUR ACTUAL MOISTURE SENSOR GPIO
#define MOISTURE_PIN 1

#define DRY_VALUE 3000
#define WET_VALUE 1200

// ================= OV3660 =================
#define PWDN_GPIO_NUM    -1
#define RESET_GPIO_NUM   -1
#define XCLK_GPIO_NUM    15
#define SIOD_GPIO_NUM     4
#define SIOC_GPIO_NUM     5
#define Y9_GPIO_NUM      16
#define Y8_GPIO_NUM      17
#define Y7_GPIO_NUM      18
#define Y6_GPIO_NUM      12
#define Y5_GPIO_NUM      10
#define Y4_GPIO_NUM       8
#define Y3_GPIO_NUM       9
#define Y2_GPIO_NUM      11
#define VSYNC_GPIO_NUM    6
#define HREF_GPIO_NUM     7
#define PCLK_GPIO_NUM    13

bool initCamera()
{
  camera_config_t config;

  config.ledc_channel = LEDC_CHANNEL_0;
  config.ledc_timer = LEDC_TIMER_0;

  config.pin_d0 = Y2_GPIO_NUM;
  config.pin_d1 = Y3_GPIO_NUM;
  config.pin_d2 = Y4_GPIO_NUM;
  config.pin_d3 = Y5_GPIO_NUM;
  config.pin_d4 = Y6_GPIO_NUM;
  config.pin_d5 = Y7_GPIO_NUM;
  config.pin_d6 = Y8_GPIO_NUM;
  config.pin_d7 = Y9_GPIO_NUM;

  config.pin_xclk = XCLK_GPIO_NUM;
  config.pin_pclk = PCLK_GPIO_NUM;
  config.pin_vsync = VSYNC_GPIO_NUM;
  config.pin_href = HREF_GPIO_NUM;

  config.pin_sccb_sda = SIOD_GPIO_NUM;
  config.pin_sccb_scl = SIOC_GPIO_NUM;

  config.pin_pwdn = PWDN_GPIO_NUM;
  config.pin_reset = RESET_GPIO_NUM;

  config.xclk_freq_hz = 20000000;
  config.pixel_format = PIXFORMAT_JPEG;
  config.frame_size = FRAMESIZE_VGA;
  config.jpeg_quality = 12;
  config.fb_location = CAMERA_FB_IN_PSRAM;
  config.grab_mode = CAMERA_GRAB_LATEST;
  config.fb_count = psramFound() ? 2 : 1;

  Serial.println("Initializing camera...");

  esp_err_t err = esp_camera_init(&config);

  if (err != ESP_OK)
  {
    Serial.print("Camera ERROR: 0x");
    Serial.println(err, HEX);
    return false;
  }

  sensor_t *sensor = esp_camera_sensor_get();

  if (sensor)
  {
    sensor->set_framesize(sensor, FRAMESIZE_VGA);
    sensor->set_quality(sensor, 12);
  }

  Serial.println("OV3660 initialized!");
  return true;
}

bool connectWiFi()
{
  WiFi.mode(WIFI_STA);
  WiFi.begin(SSID, PASSWORD);

  Serial.print("Connecting to ");
  Serial.println(SSID);

  unsigned long start = millis();

  while (WiFi.status() != WL_CONNECTED)
  {
    delay(500);
    Serial.print(".");

    if (millis() - start > 20000)
    {
      Serial.println("\nWiFi connection FAILED!");
      return false;
    }
  }

  Serial.println();
  Serial.println("WiFi connected!");
  Serial.print("Camera Node IP: ");
  Serial.println(WiFi.localIP());

  return true;
}

float readMoisture()
{
  int raw = analogRead(MOISTURE_PIN);

  Serial.print("Moisture raw ADC: ");
  Serial.println(raw);

  float moisture =
    ((float)(DRY_VALUE - raw) /
     (float)(DRY_VALUE - WET_VALUE)) * 100.0;

  moisture = constrain(moisture, 0.0, 100.0);

  Serial.print("Moisture: ");
  Serial.print(moisture, 1);
  Serial.println(" %");

  return moisture;
}

bool sendMoisture()
{
  float moisture = readMoisture();

  String url = String(MOISTURE_SERVER);
  url += "?node=";
  url += String(NODE_ID);
  url += "&value=";
  url += String(moisture, 1);

  Serial.println();
  Serial.println("Sending moisture...");
  Serial.println(url);

  HTTPClient http;
  http.begin(url);
  http.setTimeout(10000);

  int response = http.GET();

  Serial.print("Moisture HTTP response: ");
  Serial.println(response);

  bool success = (response >= 200 && response < 300);

  if (response > 0)
  {
    Serial.print("God Node: ");
    Serial.println(http.getString());
  }

  http.end();

  Serial.println(success ?
                 "MOISTURE SENT SUCCESSFULLY!" :
                 "MOISTURE SEND FAILED!");

  return success;
}

bool sendPhoto()
{
  Serial.println();
  Serial.println("Capturing photo...");

  camera_fb_t *fb = esp_camera_fb_get();

  if (!fb)
  {
    Serial.println("CAPTURE FAILED!");
    return false;
  }

  Serial.print("JPEG size: ");
  Serial.print(fb->len);
  Serial.println(" bytes");

  Serial.print("Resolution: ");
  Serial.print(fb->width);
  Serial.print(" x ");
  Serial.println(fb->height);

  HTTPClient http;

  Serial.print("Sending image to ");
  Serial.println(CAMERA_SERVER);

  http.begin(CAMERA_SERVER);
  http.addHeader("Content-Type", "image/jpeg");
  http.setTimeout(30000);

  int response = http.POST(fb->buf, fb->len);

  Serial.print("Image HTTP response: ");
  Serial.println(response);

  bool success = (response >= 200 && response < 300);

  if (response > 0)
  {
    Serial.print("God Node: ");
    Serial.println(http.getString());
  }

  http.end();
  esp_camera_fb_return(fb);

  Serial.println(success ?
                 "PHOTO SENT SUCCESSFULLY!" :
                 "PHOTO SEND FAILED!");

  return success;
}

void setup()
{
  Serial.begin(115200);
  delay(2000);

  Serial.println();
  Serial.println("====================================");
  Serial.println(" SMART FARM CAMERA NODE");
  Serial.println(" ESP32-S3 N16R8 + OV3660");
  Serial.println("====================================");

  Serial.print("NODE ID: ");
  Serial.println(NODE_ID);

  Serial.print("PSRAM: ");
  if (psramFound())
  {
    Serial.println("YES");
    Serial.print("PSRAM size: ");
    Serial.println(ESP.getPsramSize());
  }
  else
  {
    Serial.println("NO");
  }

  pinMode(MOISTURE_PIN, INPUT);
  analogReadResolution(12);

  Serial.print("Moisture GPIO: ");
  Serial.println(MOISTURE_PIN);

  if (!initCamera())
  {
    Serial.println("Camera initialization failed!");
    while (true) delay(1000);
  }

  if (!connectWiFi())
  {
    while (true) delay(1000);
  }

  // Send both datasets to the God Node
  sendMoisture();
  sendPhoto();

  Serial.println();
  Serial.println("====================================");
  Serial.println(" DATA TRANSMISSION COMPLETE");
  Serial.println("====================================");
  Serial.println("Moisture -> /moisture");
  Serial.println("Photo    -> /camera");
  Serial.println("Press RESET for another transmission.");
}

void loop()
{
  // One transmission per reset.
}
