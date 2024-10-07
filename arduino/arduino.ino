const int luzPin = 7;

void setup() {
  pinMode(luzPin, OUTPUT);
  digitalWrite(luzPin, LOW); // Luz inicialmente desligada
  Serial.begin(9600); // Comunicação com o ESP8266
}

void loop() {
  if (Serial.available()) {
    String comando = Serial.readString();

    if (comando.indexOf("ACENDER") >= 0) {
      digitalWrite(luzPin, HIGH); // Acende a luz
    } else if (comando.indexOf("APAGAR") >= 0) {
      digitalWrite(luzPin, LOW); // Apaga a luz
    }
  }
}
