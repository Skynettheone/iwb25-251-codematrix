package com.retail.pos;

import com.google.gson.Gson;
import java.net.URI;
import java.net.http.HttpClient;
import java.net.http.HttpRequest;
import java.net.http.HttpResponse;
import java.util.List;
import javafx.application.Application;
import javafx.application.Platform;
import javafx.geometry.Insets;
import javafx.geometry.Pos;
import javafx.scene.Scene;
import javafx.scene.control.Button;
import javafx.scene.control.Label;
import javafx.scene.layout.VBox;
import javafx.stage.Stage;

record Item(String productId, int quantity, double price) {}
record Transaction(String transactionId, String customerId, List<Item> items, double totalAmount) {}

public class App extends Application {

    private final HttpClient httpClient = HttpClient.newHttpClient();
    private final Gson gson = new Gson(); // JSON converter

    @Override
    public void start(Stage stage) {
        Label titleLabel = new Label("Retail POS System");
        titleLabel.setStyle("-fx-font-size: 24px; -fx-font-weight: bold;");

        Label statusLabel = new Label("Status: Idle");
        statusLabel.setStyle("-fx-font-size: 16px;");

        Button checkStatusButton = new Button("Check Backend Status");
        Button processSaleButton = new Button("Process Sample Sale");

        checkStatusButton.setOnAction(event -> {
            statusLabel.setText("Status: Checking...");
            checkBackendStatus(statusLabel);
        });

        processSaleButton.setOnAction(event -> {
            statusLabel.setText("Status: Processing sale...");
            processSampleSale(statusLabel);
        });

        VBox root = new VBox(20, titleLabel, checkStatusButton, processSaleButton, statusLabel);
        root.setAlignment(Pos.CENTER);
        root.setPadding(new Insets(25));

        Scene scene = new Scene(root, 640, 480);
        stage.setScene(scene);
        stage.setTitle("Retail POS System");
        stage.show();
    }

    private void processSampleSale(Label statusLabel) {
        List<Item> items = List.of(
            new Item("PROD-MILK", 2, 1.50),
            new Item("PROD-BREAD", 1, 2.25)
        );
        Transaction transaction = new Transaction("TXN-12345", "CUST-007", items, 5.25);

        String jsonPayload = gson.toJson(transaction);

        HttpRequest request = HttpRequest.newBuilder()
                .uri(URI.create("http://localhost:9090/api/transactions"))
                .header("Content-Type", "application/json")
                .POST(HttpRequest.BodyPublishers.ofString(jsonPayload))
                .build();

        httpClient.sendAsync(request, HttpResponse.BodyHandlers.ofString())
            .thenApply(HttpResponse::body)
            .thenAccept(responseBody -> {
                
                Platform.runLater(() -> statusLabel.setText("Sale Response: " + responseBody));
            })
            .exceptionally(error -> {
                Platform.runLater(() -> statusLabel.setText("Status: Sale failed to process."));
                System.err.println("Error processing sale: " + error.getMessage());
                return null;
            });
    }

    private void checkBackendStatus(Label statusLabel) {
        HttpRequest request = HttpRequest.newBuilder()
                .uri(URI.create("http://localhost:9090/api/status"))
                .GET()
                .build();

        httpClient.sendAsync(request, HttpResponse.BodyHandlers.ofString())
            .thenApply(HttpResponse::body)
            .thenAccept(responseBody -> {
                String status = responseBody.split(":")[1].replace("\"", "").replace("}", "").trim();
                Platform.runLater(() -> statusLabel.setText("Status: " + status));
            })
            .exceptionally(error -> {
                Platform.runLater(() -> statusLabel.setText("Status: Failed to connect."));
                System.err.println("Error connecting to backend: " + error.getMessage());
                return null;
            });
    }

    public static void main(String[] args) {
        launch();
    }
}