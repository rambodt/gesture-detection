/*
 * TinyGesture - Gesture Password Firmware
 * Target: Arduino Nano 33 BLE Sense Rev2
 *
 * Reads accelerometer + gyroscope data, runs it through an Edge Impulse
 * gesture classifier, and checks the resulting gesture sequence against
 * a predefined password.
 *
 * SETUP REQUIRED BEFORE THIS COMPILES:
 * 1. In Edge Impulse: Deployment tab -> "Arduino library" -> Build -> download the .zip
 * 2. Arduino IDE: Sketch -> Include Library -> Add .ZIP Library... -> select the downloaded file
 * 3. Install "Arduino_BMI270_BMM150" library via Library Manager (this is the correct
 *    IMU library for the Rev2 board - NOT Arduino_LSM9DS1, which is for the original Rev1)
 * 4. Replace the #include below with your actual exported header name
 *    (find it by looking inside the downloaded library's src/ folder, e.g.
 *    "EE_446_Final_Project_inferencing.h")
 * 5. Edit PASSWORD_SEQUENCE[] below to match your team's actual password
 */

// ---- REPLACE THIS with your actual Edge Impulse exported header ----
#include "EE_446_Final_Project_inferencing.h"
#include <Arduino_BMI270_BMM150.h>

// ============================================================
// CONFIGURATION - tune these based on your Model testing results
// ============================================================

// Your gesture password, in order. Must exactly match the label
// strings Edge Impulse trained on (case-sensitive).
const char* PASSWORD_SEQUENCE[] = {"Circle", "Down", "Left"};   // <-- CUSTOMIZE THIS
const int PASSWORD_LENGTH = 3;

// Minimum confidence to accept a classification as a real gesture.
// Below this, the window is treated as "uncertain" / no gesture,
// since there's no dedicated Idle class in the current label set.
const float CONFIDENCE_THRESHOLD = 0.75f;

// After a gesture is accepted, ignore new classifications for this
// long before allowing the next gesture to register. Prevents one
// physical gesture (which spans several overlapping inference windows)
// from being counted multiple times.
const unsigned long COOLDOWN_MS = 800;

// If the password sequence has been started (at least 1 correct gesture
// entered) but the next gesture doesn't arrive within this window,
// the attempt times out and resets.
const unsigned long MAX_INTER_GESTURE_MS = 4000;

// Minimum summed |accel| across a window to even bother running inference.
// Filters out "board sitting still" windows before they reach the
// classifier, since the model has no dedicated Idle class and will
// otherwise always output some gesture label for a flat/noise signal.
// TUNE THIS: hold the board still and print motion_energy for a few
// seconds to see typical "quiet" values, then set this comfortably above them.
const float MOTION_THRESHOLD = 3.0f;

// ============================================================
// STATE
// ============================================================

static float feature_buffer[EI_CLASSIFIER_DSP_INPUT_FRAME_SIZE];
static int feature_ix = 0;

int sequence_progress = 0;             // how many correct gestures entered so far
unsigned long last_gesture_time = 0;   // time of last accepted gesture
unsigned long cooldown_until = 0;      // ignore classifications until this millis()

// ============================================================
// SETUP
// ============================================================

void setup() {
  Serial.begin(115200);
  while (!Serial);

  if (!IMU.begin()) {
    Serial.println("ERROR: Failed to initialize IMU!");
    while (1);
  }

  Serial.print("Accelerometer sample rate: ");
  Serial.print(IMU.accelerationSampleRate());
  Serial.println(" Hz");
  Serial.print("Gyroscope sample rate: ");
  Serial.print(IMU.gyroscopeSampleRate());
  Serial.println(" Hz");

  if (EI_CLASSIFIER_RAW_SAMPLES_PER_FRAME != 6) {
    Serial.println("ERROR: Model expects a different number of axes than this firmware provides (expected 6: accX/Y/Z, gyrX/Y/Z).");
    while (1);
  }

  Serial.println();
  Serial.println("=== TinyGesture ready ===");
  Serial.print("Password length: ");
  Serial.println(PASSWORD_LENGTH);
  Serial.println("Waiting for gesture sequence...");
  Serial.println();
}

// ============================================================
// MAIN LOOP
// ============================================================

void loop() {
  // --- 1. Fill the feature buffer at the model's expected sample rate ---
  static unsigned long last_sample_time = 0;
  unsigned long sample_interval_ms = 1000 / EI_CLASSIFIER_FREQUENCY;

  if (millis() - last_sample_time >= sample_interval_ms) {
    last_sample_time = millis();

    float ax, ay, az, gx, gy, gz;
    if (IMU.accelerationAvailable() && IMU.gyroscopeAvailable()) {
      IMU.readAcceleration(ax, ay, az);
      IMU.readGyroscope(gx, gy, gz);

      feature_buffer[feature_ix++] = ax;
      feature_buffer[feature_ix++] = ay;
      feature_buffer[feature_ix++] = az;
      feature_buffer[feature_ix++] = gx;
      feature_buffer[feature_ix++] = gy;
      feature_buffer[feature_ix++] = gz;

      // Buffer full -> run inference
      if (feature_ix >= EI_CLASSIFIER_DSP_INPUT_FRAME_SIZE) {
        // Motion gate: skip inference entirely if the window looks
        // basically stationary (just gravity + noise).
        float motion_energy = 0;
        for (int i = 0; i < EI_CLASSIFIER_DSP_INPUT_FRAME_SIZE; i += EI_CLASSIFIER_RAW_SAMPLES_PER_FRAME) {
          motion_energy += fabs(feature_buffer[i])     // accX
                          + fabs(feature_buffer[i + 1]) // accY
                          + fabs(feature_buffer[i + 2]); // accZ
        }

        if (motion_energy >= MOTION_THRESHOLD) {
          run_inference();
        }
        // else: too still, skip classification for this window

        // Slide the window instead of clearing it entirely, so we're
        // doing overlapping continuous inference rather than waiting
        // for a full fresh window each time. Shift out the oldest
        // ~20% of samples (mirrors your Edge Impulse stride settings).
        int shift = EI_CLASSIFIER_DSP_INPUT_FRAME_SIZE / 5;
        memmove(feature_buffer, feature_buffer + shift,
                (EI_CLASSIFIER_DSP_INPUT_FRAME_SIZE - shift) * sizeof(float));
        feature_ix = EI_CLASSIFIER_DSP_INPUT_FRAME_SIZE - shift;
      }
    }
  }

  // --- 2. Handle sequence timeout ---
  if (sequence_progress > 0 && millis() - last_gesture_time > MAX_INTER_GESTURE_MS) {
    Serial.println("TIMEOUT - sequence reset");
    Serial.println("ACCESS_DENIED");
    reset_sequence();
  }
}

// ============================================================
// INFERENCE + PASSWORD LOGIC
// ============================================================

void run_inference() {
  ei::signal_t signal;
  int err = numpy::signal_from_buffer(feature_buffer, EI_CLASSIFIER_DSP_INPUT_FRAME_SIZE, &signal);
  if (err != 0) {
    Serial.print("ERROR: signal_from_buffer failed: ");
    Serial.println(err);
    return;
  }

  ei_impulse_result_t result = {0};
  err = run_classifier(&signal, &result, false);
  if (err != EI_IMPULSE_OK) {
    Serial.print("ERROR: run_classifier failed: ");
    Serial.println(err);
    return;
  }

  // Find the top predicted class and its confidence
  float best_score = 0.0f;
  const char* best_label = "";
  for (size_t ix = 0; ix < EI_CLASSIFIER_LABEL_COUNT; ix++) {
    if (result.classification[ix].value > best_score) {
      best_score = result.classification[ix].value;
      best_label = result.classification[ix].label;
    }
  }

  // Respect cooldown - ignore classifications until it expires
  if (millis() < cooldown_until) {
    return;
  }

  // Below confidence threshold -> treat as "no gesture" / uncertain
  if (best_score < CONFIDENCE_THRESHOLD) {
    return;
  }

  // We have a confident gesture - register it
  Serial.print("Gesture detected: ");
  Serial.print(best_label);
  Serial.print(" (");
  Serial.print(best_score * 100);
  Serial.println("%)");

  handle_gesture(best_label);
  cooldown_until = millis() + COOLDOWN_MS;
}

void handle_gesture(const char* label) {
  const char* expected = PASSWORD_SEQUENCE[sequence_progress];

  if (strcmp(label, expected) == 0) {
    sequence_progress++;
    last_gesture_time = millis();

    Serial.print("Correct! Progress: ");
    Serial.print(sequence_progress);
    Serial.print("/");
    Serial.println(PASSWORD_LENGTH);

    if (sequence_progress >= PASSWORD_LENGTH) {
      Serial.println();
      Serial.println("ACCESS_GRANTED");
      Serial.println();
      reset_sequence();
    }
  } else {
    Serial.print("Wrong gesture. Expected: ");
    Serial.print(expected);
    Serial.print(", got: ");
    Serial.println(label);
    Serial.println("ACCESS_DENIED");
    reset_sequence();
  }
}

void reset_sequence() {
  sequence_progress = 0;
  last_gesture_time = 0;
}
