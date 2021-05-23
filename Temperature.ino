#include <ESP8266HTTPClient.h>
#include <ESP8266WiFi.h>
#include <WiFiClient.h>
#include <ESP8266WebServer.h>
#include <Adafruit_MLX90614.h>

Adafruit_MLX90614 mlx = Adafruit_MLX90614();
String temp,postData;
void setup() {
  mlx.begin(); 
  Serial.begin(9600);                 
  WiFi.begin("Wifi_name", "Password");
  while (WiFi.status() != WL_CONNECTED) {  
  delay(500);
  Serial.println("Waiting for connection");
  }
  Serial.println("");
  Serial.print("IP Address: ");
  Serial.print(WiFi.localIP());
}

void loop() {
  HTTPClient http; 
  if (WiFi.status() == WL_CONNECTED) { //Check WiFi connection status
  Serial.print(mlx.readObjectTempC());
  Serial.println("*C");
  temp = String(mlx.readObjectTempC());
  postData = "?temp=" + temp;
  http.begin("http://enter your flask server url/temp/" + postData);
  http.addHeader("Content-Type", "text/plain");
  int httpCode = http.GET();   //Send the request   
  String payload = http.getString();                
  Serial.println(httpCode);   //Print HTTP return code
  Serial.println(payload); 
  http.end();  //Close connection
  } 
  else {
  Serial.println("Error in WiFi connection");
  }
  delay(5000);
}
