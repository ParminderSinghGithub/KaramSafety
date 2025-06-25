#include "../predict_lr.h"
#include <Wire.h>
#include <WiFi.h>
#include <WiFiUdp.h>
#include <Adafruit_BMP280.h>
#include "FastIMU.h"

#define SAMPLE_SIZE 51
#define FEATURE_DIM 10

float ax[SAMPLE_SIZE], ay[SAMPLE_SIZE], az[SAMPLE_SIZE];
float gx[SAMPLE_SIZE], gy[SAMPLE_SIZE], gz[SAMPLE_SIZE];
float mx[SAMPLE_SIZE], my[SAMPLE_SIZE], mz[SAMPLE_SIZE];
float pr[SAMPLE_SIZE];

float agg_input[40];
float scaled_agg[40];
float features[58];
float selected[47];
float final_scaled[47];

const char* ssid = "TP-Link_FC3D";
const char* password = "29061921";
const char* udpAddress = "192.168.0.102";
const int udpPortModule1 = 12344;
const int udpPortModule2 = 12345;

WiFiUDP udp1, udp2;

MPU9250 IMU;
AccelData accelData;
GyroData gyroData;
MagData magData;

Adafruit_BMP280 bmp;

int sample_idx = 0;
unsigned long last_time = 0;
float last_pressure = 0;

calData calib = {
  true,
  {-0.41, -0.29, -0.17},
  {-0.71, -0.10, -0.27},
  {139.20, 86.91, -1217.04},
  {1.11, 0.89, 1.02}
};

void setup() {
  Serial.begin(115200);
  delay(2000);

  Serial.println("=== SETUP STARTED ===");
  Wire.begin();
  Serial.println("[I2C] Wire initialized");

  WiFi.begin(ssid, password);
  Serial.print("[WiFi] Connecting");
  while (WiFi.status() != WL_CONNECTED) {
    delay(500);
    Serial.print(".");
  }
  Serial.println();
  Serial.print("[WiFi] Connected! IP: ");
  Serial.println(WiFi.localIP());

  int err = IMU.init(calib, 0x68);
  if (err != 0) {
    Serial.print("[ERROR] IMU init failed: ");
    Serial.println(err);
    while (1);
  }
  Serial.println("[IMU] MPU9250 initialized");

  if (!bmp.begin(0x76)) {
    Serial.println("[ERROR] BMP280 not found at 0x76!");
    while (1);
  }
  delay(10);
  last_pressure = bmp.readPressure();
}

void loop() {
  if (millis() - last_time >= 50) {
    IMU.update();
    IMU.getAccel(&accelData);
    IMU.getGyro(&gyroData);
    if (IMU.hasMagnetometer()) IMU.getMag(&magData);

    ax[sample_idx] = accelData.accelX;
    ay[sample_idx] = accelData.accelY;
    az[sample_idx] = accelData.accelZ;
    gx[sample_idx] = gyroData.gyroX;
    gy[sample_idx] = gyroData.gyroY;
    gz[sample_idx] = gyroData.gyroZ;
    mx[sample_idx] = magData.magX;
    my[sample_idx] = magData.magY;
    mz[sample_idx] = magData.magZ;

    float current_pressure = bmp.readPressure();
    pr[sample_idx] = current_pressure - last_pressure;
    last_pressure = current_pressure;

    sample_idx++;
    last_time = millis();
  }

  if (sample_idx >= SAMPLE_SIZE) {
    float input_batch[10][51];
    for (int i = 0; i < SAMPLE_SIZE; i++) {
      input_batch[0][i] = ax[i]; input_batch[1][i] = ay[i]; input_batch[2][i] = az[i];
      input_batch[3][i] = gx[i]; input_batch[4][i] = gy[i]; input_batch[5][i] = gz[i];
      input_batch[6][i] = mx[i]; input_batch[7][i] = my[i]; input_batch[8][i] = mz[i];
      input_batch[9][i] = pr[i];
    }

    for (int i = 0; i < 10; ++i) {
      float sum = 0, min_v = input_batch[i][0], max_v = input_batch[i][0];
      for (int j = 0; j < SAMPLE_SIZE; ++j) {
        float val = input_batch[i][j];
        sum += val;
        if (val < min_v) min_v = val;
        if (val > max_v) max_v = val;
      }
      float mean = sum / SAMPLE_SIZE;
      float sq_sum = 0;
      for (int j = 0; j < SAMPLE_SIZE; ++j)
        sq_sum += (input_batch[i][j] - mean) * (input_batch[i][j] - mean);
      float std = sqrt(sq_sum / SAMPLE_SIZE);

      agg_input[i * 4 + 0] = mean;
      agg_input[i * 4 + 1] = std;
      agg_input[i * 4 + 2] = min_v;
      agg_input[i * 4 + 3] = max_v;
    }

    scale_aggregated(agg_input, scaled_agg);
    extract_all_features(input_batch, features + 40);
    for (int i = 0; i < 40; ++i) features[i] = scaled_agg[i];

    select_features(features, selected);
    scale_final(selected, final_scaled);
    int pred = predict_class(final_scaled);

    Serial.print("Predicted class (Module 1 & 2): ");
    Serial.println(pred);

    char msg[50];
    snprintf(msg, sizeof(msg), "Prediction: %d", pred);

    udp1.beginPacket(udpAddress, udpPortModule1);
    udp1.write((const uint8_t*)msg, strlen(msg));
    udp1.endPacket();

    udp2.beginPacket(udpAddress, udpPortModule2);
    udp2.write((const uint8_t*)msg, strlen(msg));
    udp2.endPacket();

    sample_idx = 0;
  }
}
