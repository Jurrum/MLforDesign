#include <LSM6DS3.h>

#include "LSM6DS3.h"
#include "Wire.h"

//Create a instance of class LSM6DS3
LSM6DS3 myIMU(I2C_MODE, 0x6A);    //I2C device address 0x6A

void setup() {
    Serial.begin(9600);
    
    while (!Serial);
    //Call .begin() to configure the IMUs
    if (myIMU.begin() != 0) {
        Serial.println("Device error");
    } else {
        Serial.println("Device OK!");
    }
}

void loop() {

    unsigned long timestamp = millis(); // Get the current time in milliseconds

    //Accelerometer
    Serial.print(timestamp);
    Serial.print(",");
    Serial.print(myIMU.readFloatAccelX(), 20);
    Serial.print(",");
    Serial.print(myIMU.readFloatAccelY(), 20);
    Serial.print(",");
    Serial.println(myIMU.readFloatAccelZ(), 20);

    //Gyroscope
    Serial.print(timestamp);
    Serial.print(",");
    Serial.print(myIMU.readFloatGyroX(), 20);
    Serial.print(",");
    Serial.print(myIMU.readFloatGyroY(), 20);
    Serial.print(",");
    Serial.println(myIMU.readFloatGyroZ(), 20);


    delay(10);
}
