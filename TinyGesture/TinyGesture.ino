
#include <EE_446_Final_Project_inferencing.h>
#include <Arduino_BMI270_BMM150.h>
#include <Arduino_LPS22HB.h>
#include <Arduino_HS300x.h>
#include <Arduino_APDS9960.h>

enum sensor_status {
    NOT_USED = -1,
    NOT_INIT,
    INIT,
    SAMPLED
};

typedef struct {
    const char *name;
    float *value;
    uint8_t (*poll_sensor)(void);
    bool (*init_sensor)(void);
    sensor_status status;
} eiSensors;

#define CONVERT_G_TO_MS2    9.80665f
#define MAX_ACCEPTED_RANGE  2.0f

#define N_SENSORS 18
#define GESTURE_CONFIDENCE 0.70f
#define GESTURE_COOLDOWN_MS 1000

float ei_get_sign(float number);

bool init_IMU(void);
bool init_HTS(void);
bool init_BARO(void);
bool init_APDS(void);

uint8_t poll_acc(void);
uint8_t poll_gyr(void);
uint8_t poll_mag(void);
uint8_t poll_HTS(void);
uint8_t poll_BARO(void);
uint8_t poll_APDS_color(void);
uint8_t poll_APDS_proximity(void);
uint8_t poll_APDS_gesture(void);

static const bool debug_nn = false;

static float data[N_SENSORS];

static bool ei_connect_fusion_list(const char *input_list);

static int8_t fusion_sensors[N_SENSORS];
static int fusion_ix = 0;

eiSensors sensors[] = {
  {"accX", &data[0], &poll_acc, &init_IMU, NOT_USED},
  {"accY", &data[1], &poll_acc, &init_IMU, NOT_USED},
  {"accZ", &data[2], &poll_acc, &init_IMU, NOT_USED},

  {"gyrX", &data[3], &poll_gyr, &init_IMU, NOT_USED},
  {"gyrY", &data[4], &poll_gyr, &init_IMU, NOT_USED},
  {"gyrZ", &data[5], &poll_gyr, &init_IMU, NOT_USED},

  {"magX", &data[6], &poll_mag, &init_IMU, NOT_USED},
  {"magY", &data[7], &poll_mag, &init_IMU, NOT_USED},
  {"magZ", &data[8], &poll_mag, &init_IMU, NOT_USED},

  {"temperature", &data[9], &poll_HTS, &init_HTS, NOT_USED},
  {"humidity", &data[10], &poll_HTS, &init_HTS, NOT_USED},

  {"pressure", &data[11], &poll_BARO, &init_BARO, NOT_USED},

  {"red", &data[12], &poll_APDS_color, &init_APDS, NOT_USED},
  {"green", &data[13], &poll_APDS_color, &init_APDS, NOT_USED},
  {"blue", &data[14], &poll_APDS_color, &init_APDS, NOT_USED},
  {"brightness", &data[15], &poll_APDS_color, &init_APDS, NOT_USED},

  {"proximity", &data[16], &poll_APDS_proximity, &init_APDS, NOT_USED},

  {"gesture", &data[17], &poll_APDS_gesture, &init_APDS, NOT_USED}
};


enum PasswordState {
  IDLE,
  WAITING_FOR_GESTURE,
  PASSWORD_CORRECT,
  PASSWORD_INCORRECT, 
  PASSWORD_UNLOCKED
};

PasswordState passwordState = IDLE;

const char* password[] = {
  "Left1",
  "Up1",
  "Right1",
  "Down1"
};

const int PASSWORD_LENGTH = 4;

int passwordIndex = 0;
unsigned long lastGestureTime = 0;

void setup()
{
  Serial.begin(115200);

  while (!Serial);

  Serial.println();
  Serial.println("==============================");
  Serial.println("Edge Impulse Gesture Password");
  Serial.println("==============================");
  Serial.println();

  if (ei_connect_fusion_list(EI_CLASSIFIER_FUSION_AXES_STRING) == false) {
      ei_printf(
          "ERR: Errors in sensor list detected\r\n"
      );
      return;
  }

  for (int i = 0; i < fusion_ix; i++) {
      if (sensors[fusion_sensors[i]].status == NOT_INIT) {
          sensors[fusion_sensors[i]].status =
              (sensor_status)
              sensors[fusion_sensors[i]].init_sensor();
          if (!sensors[fusion_sensors[i]].status) {
              ei_printf(
                  "%s axis sensor initialization failed.\r\n",
                  sensors[fusion_sensors[i]].name
              );
          }
          else {
              ei_printf(
                  "%s axis sensor initialization successful.\r\n",
                  sensors[fusion_sensors[i]].name
              );
          }
      }
  }
  if (EI_CLASSIFIER_RAW_SAMPLES_PER_FRAME != fusion_ix) {
      ei_printf(
          "ERR: Sensors don't match the sensors required "
          "in the model\r\n"
      );
      return;
  }
  enterIdle();
}

void enterIdle(){
  passwordState = IDLE;
  passwordIndex = 0;

  Serial.println();
  Serial.println("----------------------");
  Serial.println("IDLE");
  Serial.println("System is waiting.");
  Serial.println("Perform LEFT to begin.");
  Serial.println("----------------------");
}

void startPassword(){
  passwordState = WAITING_FOR_GESTURE;
  passwordIndex = 0;

  Serial.println();
  Serial.println("PASSWORD ENTRY STARTED");

  Serial.print("Expected: ");
  Serial.println(password[passwordIndex]);
}

void processGesture(const char* gesture) {
  if (passwordState == IDLE) {
      if (strcmp(gesture, "Left1") == 0) {
          Serial.println();
          Serial.println("LEFT detected.");

          startPassword();
          passwordIndex = 1;

          Serial.print("Next expected: ");
          Serial.println(password[passwordIndex]);
      }
      return;
  }
  if (passwordState == WAITING_FOR_GESTURE) {
    Serial.print("Detected: ");
    Serial.println(gesture);

    Serial.print("Expected: ");
    Serial.println(password[passwordIndex]);

    if (strcmp(gesture, password[passwordIndex]) == 0) {
        Serial.println("CORRECT!");
        passwordIndex++;
        if (passwordIndex >= PASSWORD_LENGTH) {
          passwordState = PASSWORD_CORRECT;

          Serial.println();
          Serial.println("=================");
          Serial.println("PASSWORD CORRECT!");
          Serial.println("=================");
          Serial.println();
          delay(2000);
          passwordState = PASSWORD_UNLOCKED;
        }
        else {
          Serial.print("Next expected: ");
          Serial.println(password[passwordIndex]);
        }
    }
    else {
      passwordState = PASSWORD_INCORRECT;

      Serial.println();
      Serial.println("===================");
      Serial.println("PASSWORD INCORRECT");
      Serial.println("==================");
      Serial.println();

      delay(2000);
      enterIdle();
    }
  }
}

void loop() {
  if (passwordState == PASSWORD_UNLOCKED) {
        return;
  }

  ei_printf("\nStarting inferencing in 1 second...\r\n");
  delay(1000);
  if (millis() - lastGestureTime < GESTURE_COOLDOWN_MS) {
      return;
  }
  ei_printf("Sampling...\r\n");
  float buffer[EI_CLASSIFIER_DSP_INPUT_FRAME_SIZE] = { 0 };
  for (size_t ix = 0; ix < EI_CLASSIFIER_DSP_INPUT_FRAME_SIZE; ix += EI_CLASSIFIER_RAW_SAMPLES_PER_FRAME) {
    int64_t next_tick =
        (int64_t)micros() +
        ((int64_t)EI_CLASSIFIER_INTERVAL_MS * 1000);
    for (int i = 0; i < fusion_ix; i++) {
        if (sensors[fusion_sensors[i]].status== INIT) {
            sensors[fusion_sensors[i]].poll_sensor();
            sensors[fusion_sensors[i]].status = SAMPLED;
          }
        if (sensors[fusion_sensors[i]].status == SAMPLED) {
            buffer[ix + i] =*sensors[fusion_sensors[i]].value;
            sensors[fusion_sensors[i]].status = INIT;
        }
    }
    int64_t wait_time = next_tick - (int64_t)micros();
    if (wait_time > 0) {
      delayMicroseconds(wait_time);
    }
  }
  signal_t signal;
  int err = numpy::signal_from_buffer(buffer, EI_CLASSIFIER_DSP_INPUT_FRAME_SIZE, &signal);
  if (err != 0) {
      ei_printf(
          "ERR creating signal: %d\r\n",
          err
      );
      return;
  }
  ei_impulse_result_t result = { 0 };
  err = run_classifier(&signal, &result, debug_nn);
  if (err != EI_IMPULSE_OK) {
      ei_printf("ERR running classifier: %d\r\n", err);
      return;
  }
  float highestConfidence = 0.0f;
  int bestGestureIndex = -1;
  for (size_t ix = 0;ix < EI_CLASSIFIER_LABEL_COUNT;ix++) {
    float confidence =
        result.classification[ix].value;
    ei_printf(
        "%s: %.5f\r\n",
        result.classification[ix].label,
        confidence
    );
    if (confidence > highestConfidence) {

        highestConfidence = confidence;

        bestGestureIndex = ix;
    }
  }
  if (
      bestGestureIndex >= 0 &&
      highestConfidence >= GESTURE_CONFIDENCE
  ) {
    const char* detectedGesture =
        result.classification[
            bestGestureIndex
        ].label;


    Serial.println();
    Serial.print("GESTURE DETECTED: ");
    Serial.print(detectedGesture);

    Serial.print(" (");
    Serial.print(
        highestConfidence * 100.0,
        1
    );
    Serial.println("%)");
    processGesture(detectedGesture);
    lastGestureTime = millis();
  }
  else {
      Serial.println(
          "No confident gesture detected."
      );
  }


#if EI_CLASSIFIER_HAS_ANOMALY == 1

  ei_printf(
      "Anomaly: %.3f\r\n",
      result.anomaly
  );

#endif
}

#if !defined(EI_CLASSIFIER_SENSOR) || \
    (EI_CLASSIFIER_SENSOR != EI_CLASSIFIER_SENSOR_FUSION && \
     EI_CLASSIFIER_SENSOR != EI_CLASSIFIER_SENSOR_ACCELEROMETER)

#error "Invalid model for current sensor"

#endif

static int8_t ei_find_axis(char *axis_name){
  int ix;

  for (ix = 0; ix < N_SENSORS; ix++) {

      if (strstr(axis_name, sensors[ix].name)) {
          return ix;
      }
  }

  return -1;
}


static bool ei_connect_fusion_list(const char *input_list) {
  char *buff;

  bool is_fusion = false;


  char *input_string =
      (char *)ei_malloc(
          strlen(input_list) + 1
      );


  if (input_string == NULL) {
      return false;
  }


  memset(
      input_string,
      0,
      strlen(input_list) + 1
  );


  strncpy(
      input_string,
      input_list,
      strlen(input_list)
  );


  memset(
      fusion_sensors,
      0,
      N_SENSORS
  );

  fusion_ix = 0;


  buff = strtok(
      input_string,
      "+"
  );


  while (buff != NULL) {

      int8_t found_axis = 0;

      is_fusion = false;

      found_axis =
          ei_find_axis(buff);


      if (found_axis >= 0) {

          if (fusion_ix < N_SENSORS) {

              fusion_sensors[fusion_ix++] =
                  found_axis;

              sensors[found_axis].status =
                  NOT_INIT;
          }

          is_fusion = true;
      }


      buff = strtok(
          NULL,
          "+ "
      );
  }


  ei_free(input_string);

  return is_fusion;
}
float ei_get_sign(float number){
  return (number >= 0.0)
      ? 1.0
      : -1.0;
}


bool init_IMU(void){
  static bool init_status = false;

  if (!init_status) {
      init_status = IMU.begin();
  }

  return init_status;
}


uint8_t poll_acc(void){
  if (IMU.accelerationAvailable()) {

      IMU.readAcceleration(
          data[0],
          data[1],
          data[2]
      );


      for (int i = 0; i < 3; i++) {

          if (
              fabs(data[i])
              > MAX_ACCEPTED_RANGE
          ) {

              data[i] =
                  ei_get_sign(data[i])
                  * MAX_ACCEPTED_RANGE;
          }
      }


      data[0] *= CONVERT_G_TO_MS2;
      data[1] *= CONVERT_G_TO_MS2;
      data[2] *= CONVERT_G_TO_MS2;
  }

  return 0;
}


uint8_t poll_gyr(void){
  if (IMU.gyroscopeAvailable()) {

      IMU.readGyroscope(
          data[3],
          data[4],
          data[5]
      );
  }

  return 0;
}


uint8_t poll_mag(void){
  if (IMU.magneticFieldAvailable()) {

      IMU.readMagneticField(
          data[6],
          data[7],
          data[8]
      );
  }

  return 0;
}


bool init_HTS(void){
  static bool init_status = false;

  if (!init_status) {
      init_status = HS300x.begin();
  }

  return init_status;
}


bool init_BARO(void){
  static bool init_status = false;

  if (!init_status) {
      init_status = BARO.begin();
  }

  return init_status;
}


bool init_APDS(void){
  static bool init_status = false;
  if (!init_status) {
      init_status = APDS.begin();
  }
  return init_status;
}


uint8_t poll_HTS(void){
  data[9] = HS300x.readTemperature();
  data[10] = HS300x.readHumidity();

  return 0;
}


uint8_t poll_BARO(void){
  data[11] = BARO.readPressure();

  return 0;
}


uint8_t poll_APDS_color(void){
  int temp_data[4];
  if (APDS.colorAvailable()) {
      APDS.readColor(
          temp_data[0],
          temp_data[1],
          temp_data[2],
          temp_data[3]
      );
      data[12] = temp_data[0];
      data[13] = temp_data[1];
      data[14] = temp_data[2];
      data[15] = temp_data[3];
  }
  return 0;
}


uint8_t poll_APDS_proximity(void){
  if (APDS.proximityAvailable()) {
      data[16] =
          (float)APDS.readProximity();
  }
  return 0;
}


uint8_t poll_APDS_gesture(void){
  if (APDS.gestureAvailable()) {
      data[17] =
          (float)APDS.readGesture();
  }

  return 0;
}