#include <ESP8266WiFi.h>
#include <ESP8266WebServer.h>

// Configurar Wi-Fi
const char* ssid = "Santos";
const char* password = "rocha2024";

// Configurar o servidor web
ESP8266WebServer server(80);

void setup() {
  Serial.begin(9600); // Comunicação com o Arduino Mega
  WiFi.begin(ssid, password);

  while (WiFi.status() != WL_CONNECTED) {
    delay(1000);
    Serial.println("Conectando ao WiFi...");
  }

  Serial.println("Conectado ao WiFi");
  Serial.println(WiFi.localIP());

  server.on("/acender", []() {
    Serial.println("ACENDER");  // Enviar comando para o Mega
    server.send(200, "text/plain", "Luz acesa");
  });

  server.on("/apagar", []() {
    Serial.println("APAGAR");  // Enviar comando para o Mega
    server.send(200, "text/plain", "Luz apagada");
  });

  server.begin();
}

void loop() {
  server.handleClient();
}
