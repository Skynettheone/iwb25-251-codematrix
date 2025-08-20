package com.retail.pos;

import java.net.URI;
import java.net.http.HttpClient;
import java.net.http.HttpRequest;
import java.net.http.HttpResponse;
import java.util.ArrayList;
import java.util.List;
import java.util.UUID;

import com.google.gson.Gson;
import com.google.gson.reflect.TypeToken;

import javafx.application.Application;
import javafx.application.Platform;
import javafx.collections.FXCollections;
import javafx.collections.ObservableList;
import javafx.geometry.Insets;
import javafx.geometry.Pos;
import javafx.scene.Scene;
import javafx.scene.control.Button;
import javafx.scene.control.Label;
import javafx.scene.control.Separator;
import javafx.scene.control.TableColumn;
import javafx.scene.control.TableView;
import javafx.scene.control.TextField;
import javafx.scene.layout.BorderPane;
import javafx.scene.layout.HBox;
import javafx.scene.layout.Pane;
import javafx.scene.layout.Priority;
import javafx.scene.layout.VBox;
import javafx.stage.Stage;

record Item(String product_id, int quantity, double price_at_sale) {}
record Transaction(String transaction_id, String customer_id, List<Item> items, double total_amount) {}
record Product(String product_id, String name, String category, double price) {}
record Customer(String customer_id, String name, String email, String phone_number) {}

public class App extends Application {

    private final HttpClient httpClient = HttpClient.newHttpClient();
    private final Gson gson = new Gson();

    private final ObservableList<Product> products = FXCollections.observableArrayList();
    private final ObservableList<Item> cart = FXCollections.observableArrayList();

    // Auth state for POS (cashiers)
    private String authToken = null;
    private String authRole = null;
    private String authUser = null;

    @Override
    public void start(Stage stage) {
        BorderPane root = new BorderPane();
        root.setPadding(new Insets(12));


    Label status = new Label("Ready");
    Button refreshBtn = new Button("Refresh Products");
    Button healthBtn = new Button("Check Backend");
    Button loginBtn = new Button("Login");
    Button logoutBtn = new Button("Logout"); logoutBtn.setDisable(true);
    Label userLbl = new Label("");
    HBox topBar = new HBox(10, refreshBtn, healthBtn, loginBtn, logoutBtn, userLbl, status);
        topBar.setAlignment(Pos.CENTER_LEFT);
        topBar.setPadding(new Insets(8));
        root.setTop(topBar);


        TableView<Product> productTable = new TableView<>(products);

    productTable.getColumns().add(pcol("ID", p -> p.product_id()));
    productTable.getColumns().add(pcol("Name", p -> p.name()));
    productTable.getColumns().add(pcol("Cat", p -> p.category()));
    productTable.getColumns().add(pcol("Price", p -> String.format("%.2f", p.price())));
    productTable.setPrefWidth(540);

        TextField productSearch = new TextField();
        productSearch.setPromptText("Search products by name/category/id...");
        productSearch.textProperty().addListener((obs, old, val) -> {
            String q = val == null ? "" : val.toLowerCase();

            productTable.setItems(products.filtered(p ->
                p.product_id().toLowerCase().contains(q) ||
                p.name().toLowerCase().contains(q) ||
                (p.category() != null && p.category().toLowerCase().contains(q))
            ));
        });

    TextField quickId = new TextField(); quickId.setPromptText("Scan/Type Product ID");
    Button quickAdd = new Button("Add");
    HBox quickRow = new HBox(8, quickId, quickAdd); quickRow.setAlignment(Pos.CENTER_LEFT);
    VBox left = new VBox(8, new Label("Products"), productSearch, productTable, quickRow, productCrudPane(status));
        VBox.setVgrow(productTable, Priority.ALWAYS);
        root.setLeft(left);


        TableView<Item> cartTable = new TableView<>(cart);
    cartTable.getColumns().add(icol("Product", i -> i.product_id()));
    cartTable.getColumns().add(icol("Qty", i -> String.valueOf(i.quantity())));
    cartTable.getColumns().add(icol("Price", i -> String.format("%.2f", i.price_at_sale())));
        cartTable.setPrefWidth(360);

        TextField qtyField = new TextField("1");
        qtyField.setPrefWidth(60);
        Button addToCart = new Button("Add to Cart");
    Button removeItem = new Button("Remove");
    Button clearCart = new Button("Clear Cart");
    HBox addBar = new HBox(8, new Label("Qty"), qtyField, addToCart, removeItem, clearCart);
        addBar.setAlignment(Pos.CENTER_LEFT);

    Label subLbl = new Label("Subtotal: 0.00");
    TextField discountField = new TextField("0"); discountField.setPrefWidth(60);
    TextField taxField = new TextField("0"); taxField.setPrefWidth(60);
    HBox calcRow = new HBox(10, new Label("Disc %"), discountField, new Label("Tax %"), taxField);
    Label totalLbl = new Label("Total: 0.00");
        Button checkoutBtn = new Button("Checkout");
    VBox center = new VBox(8, new Label("Cart"), cartTable, addBar, calcRow, new Separator(), subLbl, totalLbl, checkoutBtn);
        VBox.setVgrow(cartTable, Priority.ALWAYS);
        root.setCenter(center);


        TextField custId = new TextField(); custId.setPromptText("Customer ID");
        TextField custName = new TextField(); custName.setPromptText("Name");
        TextField custEmail = new TextField(); custEmail.setPromptText("Email");
        TextField custPhone = new TextField(); custPhone.setPromptText("Phone");
    Button lookupCust = new Button("Lookup");
    Button createCust = new Button("Create Customer");

    lookupCust.setOnAction(e -> lookupCustomer(status, custId, custName, custEmail, custPhone));

    VBox right = new VBox(8, new Label("Customer"), custId, custName, custEmail, custPhone, new HBox(8, lookupCust, createCust));
        right.setPrefWidth(260);
        root.setRight(right);


        refreshBtn.setOnAction(e -> loadProducts(status));
        healthBtn.setOnAction(e -> checkHealth(status));
        loginBtn.setOnAction(e -> doLogin(status, userLbl, logoutBtn));
        logoutBtn.setOnAction(e -> {
            doLogout(status, userLbl, logoutBtn);
        });
        Runnable recalc = () -> {
            double subtotal = cart.stream().mapToDouble(i -> i.quantity() * i.price_at_sale()).sum();
            subLbl.setText("Subtotal: " + String.format("%.2f", subtotal));
            double disc = safeParseDouble(discountField.getText(), 0) / 100.0;
            double tax = safeParseDouble(taxField.getText(), 0) / 100.0;
            double afterDisc = subtotal * (1 - Math.max(0, Math.min(1, disc)));
            double total = afterDisc * (1 + Math.max(0, Math.min(1, tax)));
            totalLbl.setText("Total: " + String.format("%.2f", total));
        };

        addToCart.setOnAction(e -> {
            Product sel = productTable.getSelectionModel().getSelectedItem();
            if (sel == null) { status.setText("Select a product"); return; }
            int qty = Math.max(1, safeParseInt(qtyField.getText(), 1));
            cart.add(new Item(sel.product_id(), qty, sel.price()));
            recalc.run();
        });

        productTable.setOnMouseClicked(ev -> {
            if (ev.getClickCount() == 2) {
                Product sel = productTable.getSelectionModel().getSelectedItem();
                if (sel != null) { cart.add(new Item(sel.product_id(), 1, sel.price())); recalc.run(); }
            }
        });

        removeItem.setOnAction(e -> { Item it = cartTable.getSelectionModel().getSelectedItem(); if (it != null) { cart.remove(it); recalc.run(); }});

        clearCart.setOnAction(e -> { cart.clear(); recalc.run(); });

        discountField.textProperty().addListener((o,old,v)-> recalc.run());
        taxField.textProperty().addListener((o,old,v)-> recalc.run());

        quickAdd.setOnAction(e -> addProductById(quickId.getText(), 1, status, recalc));
        quickId.setOnAction(e -> addProductById(quickId.getText(), 1, status, recalc));
        checkoutBtn.setOnAction(e -> checkout(status, custId.getText().isBlank() ? "WALKIN" : custId.getText(), totalLbl));
        createCust.setOnAction(e -> createCustomer(status, custId.getText(), custName.getText(), custEmail.getText(), custPhone.getText()));


        Scene scene = new Scene(root, 1200, 700);
        stage.setTitle("Retail POS");
        stage.setScene(scene);
        stage.show();


    Platform.runLater(() -> doLogin(status, userLbl, logoutBtn));
    }

    private TableColumn<Product, String> pcol(String name, java.util.function.Function<Product, String> f) {
        TableColumn<Product, String> c = new TableColumn<>(name);
        c.setCellValueFactory(data -> new javafx.beans.property.SimpleStringProperty(f.apply(data.getValue())));
        c.setPrefWidth(120);
        return c;
    }
    private TableColumn<Item, String> icol(String name, java.util.function.Function<Item, String> f) {
        TableColumn<Item, String> c = new TableColumn<>(name);
        c.setCellValueFactory(data -> new javafx.beans.property.SimpleStringProperty(f.apply(data.getValue())));
        c.setPrefWidth(100);
        return c;
    }

    private int safeParseInt(String s, int def) {
    try { return Integer.parseInt(s.trim()); } catch (NumberFormatException e) { return def; }
    }

    private double safeParseDouble(String s, double def) {
        try { return Double.parseDouble(s.trim()); } catch (NumberFormatException e) { return def; }
    }

    private void addProductById(String pid, int qty, Label status, Runnable recalc) {
        if (pid == null || pid.isBlank()) { return; }
        Product match = products.stream().filter(p -> p.product_id().equalsIgnoreCase(pid.trim())).findFirst().orElse(null);
        if (match != null) {
            cart.add(new Item(match.product_id(), Math.max(1, qty), match.price()));
            recalc.run();
        } else {
            status.setText("Product not found: " + pid);
        }
    }

    private com.google.gson.JsonObject parseJsonObjectOrNull(String body) {
        try {
            var elem = com.google.gson.JsonParser.parseString(body);
            return elem.isJsonObject() ? elem.getAsJsonObject() : null;
        } catch (RuntimeException e) {
            return null;
        }
    }

    private Pane productCrudPane(Label status) {
        TextField id = new TextField(); id.setPromptText("Product ID");
        TextField name = new TextField(); name.setPromptText("Name");
        TextField category = new TextField(); category.setPromptText("Category");
        TextField price = new TextField(); price.setPromptText("Price");
        Button create = new Button("Add");
        Button update = new Button("Update");
        Button remove = new Button("Delete");
        HBox buttons = new HBox(8, create, update, remove);
        VBox box = new VBox(6, new Label("Manage Product"), id, name, category, price, buttons);
        box.setPadding(new Insets(8));

        create.setOnAction(e -> {
            try {
                double p = Double.parseDouble(price.getText());
                Product prod = new Product(id.getText(), name.getText(), category.getText(), p);
                String payload = gson.toJson(prod);
                HttpRequest.Builder builder = HttpRequest.newBuilder(URI.create("http://localhost:9090/api/products"))
                    .header("Content-Type", "application/json");
                if (authToken != null) builder.header("Authorization", "Bearer " + authToken);
                HttpRequest req = builder
                    .POST(HttpRequest.BodyPublishers.ofString(payload)).build();
                httpClient.sendAsync(req, HttpResponse.BodyHandlers.ofString())
                    .thenApply(HttpResponse::body)
                    .thenAccept(body -> Platform.runLater(() -> { status.setText("Product added"); loadProducts(status);} ))
                    .exceptionally(err -> { Platform.runLater(() -> status.setText("Add failed")); return null; });
            } catch (NumberFormatException ex) { status.setText("Invalid price"); }
        });

        update.setOnAction(e -> {
            try {
                double p = Double.parseDouble(price.getText());
                Product prod = new Product(id.getText(), name.getText(), category.getText(), p);
                String payload = gson.toJson(prod);
                HttpRequest.Builder builder = HttpRequest.newBuilder(URI.create("http://localhost:9090/api/products/" + id.getText()))
                    .header("Content-Type", "application/json");
                if (authToken != null) builder.header("Authorization", "Bearer " + authToken);
                HttpRequest req = builder
                    .PUT(HttpRequest.BodyPublishers.ofString(payload)).build();
                httpClient.sendAsync(req, HttpResponse.BodyHandlers.ofString())
                    .thenApply(HttpResponse::body)
                    .thenAccept(body -> Platform.runLater(() -> { status.setText("Product updated"); loadProducts(status);} ))
                    .exceptionally(err -> { Platform.runLater(() -> status.setText("Update failed")); return null; });
            } catch (NumberFormatException ex) { status.setText("Invalid price"); }
        });

        remove.setOnAction(e -> {
            HttpRequest.Builder builder = HttpRequest.newBuilder(URI.create("http://localhost:9090/api/products/" + id.getText()));
            if (authToken != null) builder.header("Authorization", "Bearer " + authToken);
            HttpRequest req = builder.DELETE().build();
            httpClient.sendAsync(req, HttpResponse.BodyHandlers.ofString())
                .thenApply(HttpResponse::body)
                .thenAccept(body -> Platform.runLater(() -> { status.setText("Product deleted"); loadProducts(status);} ))
                .exceptionally(err -> { Platform.runLater(() -> status.setText("Delete failed")); return null; });
        });

        return box;
    }

    private void loadProducts(Label status) {
        status.setText("Loading products...");
    HttpRequest.Builder b = HttpRequest.newBuilder(URI.create("http://localhost:9090/api/products"));
    if (authToken != null) b.header("Authorization", "Bearer " + authToken);
    HttpRequest req = b.GET().build();
        httpClient.sendAsync(req, HttpResponse.BodyHandlers.ofString())
            .thenApply(HttpResponse::body)
            .thenAccept(body -> {
                try {
                    // backend returns either raw array or {status,data}
                    List<Product> list;
                    if (body.trim().startsWith("{")) {
                        var tree = com.google.gson.JsonParser.parseString(body).getAsJsonObject();
                        if (tree.has("data")) {
                            list = gson.fromJson(tree.get("data"), new TypeToken<List<Product>>(){}.getType());
                        } else {
                            list = gson.fromJson(tree, new TypeToken<List<Product>>(){}.getType());
                        }
                    } else {
                        list = gson.fromJson(body, new TypeToken<List<Product>>(){}.getType());
                    }
                    List<Product> finalList = list == null ? List.of() : list;
                    Platform.runLater(() -> {
                        products.setAll(finalList);
                        status.setText("Loaded " + finalList.size() + " products");
                    });
                } catch (RuntimeException ex) {
                    Platform.runLater(() -> status.setText("Failed to parse products"));
                }
            })
            .exceptionally(err -> { Platform.runLater(() -> status.setText("Failed to load products")); return null; });
    }

    private void lookupCustomer(Label status, TextField custId, TextField custName, TextField custEmail, TextField custPhone) {
        String qTmp = custId.getText();
        if (qTmp == null || qTmp.isBlank()) {
            qTmp = custPhone.getText();
        }
        final String query = qTmp;
        final String qLower = query == null ? "" : query.toLowerCase();
        status.setText("Looking up customer...");
    HttpRequest.Builder b = HttpRequest.newBuilder(URI.create("http://localhost:9090/api/customers"));
    if (authToken != null) b.header("Authorization", "Bearer " + authToken);
    HttpRequest req = b.GET().build();
        httpClient.sendAsync(req, HttpResponse.BodyHandlers.ofString())
            .thenApply(HttpResponse::body)
            .thenAccept(body -> {
                try {
                    List<Customer> list;
                    if (body.trim().startsWith("{")) {
                        var tree = com.google.gson.JsonParser.parseString(body).getAsJsonObject();
                        if (tree.has("data")) {
                            list = gson.fromJson(tree.get("data"), new TypeToken<List<Customer>>(){}.getType());
                        } else {
                            list = gson.fromJson(tree, new TypeToken<List<Customer>>(){}.getType());
                        }
                    } else {
                        list = gson.fromJson(body, new TypeToken<List<Customer>>(){}.getType());
                    }
                    if (list == null) list = List.of();
                    // simple lookup by id, name, or phone contains
                    Customer match = null;
                    for (Customer c : list) {
                        if ((c.customer_id()!=null && query != null && c.customer_id().equalsIgnoreCase(query)) ||
                            (c.phone_number()!=null && c.phone_number().toLowerCase().contains(qLower)) ||
                            (c.name()!=null && c.name().toLowerCase().contains(qLower))) {
                            match = c; break;
                        }
                    }
                    final Customer found = match;
                    Platform.runLater(() -> {
                        if (found != null) {
                            custId.setText(found.customer_id());
                            custName.setText(found.name());
                            custEmail.setText(found.email());
                            custPhone.setText(found.phone_number());
                            status.setText("Customer found and filled");
                        } else {
                            status.setText("No matching customer");
                        }
                    });
                } catch (RuntimeException ex) {
                    Platform.runLater(() -> status.setText("Failed to parse customers"));
                }
            })
            .exceptionally(err -> { Platform.runLater(() -> status.setText("Lookup failed")); return null; });
    }

    private void checkout(Label status, String customerId, Label totalLbl) {
        if (cart.isEmpty()) { status.setText("Cart empty"); return; }
        double total = cart.stream().mapToDouble(i -> i.quantity() * i.price_at_sale()).sum();
        Transaction tx = new Transaction("TXN-" + UUID.randomUUID(), customerId, new ArrayList<>(cart), total);
        String payload = gson.toJson(tx);
    HttpRequest.Builder b = HttpRequest.newBuilder(URI.create("http://localhost:9090/api/transactions"))
        .header("Content-Type", "application/json");
    if (authToken != null) b.header("Authorization", "Bearer " + authToken);
    HttpRequest req = b
                .POST(HttpRequest.BodyPublishers.ofString(payload)).build();
        httpClient.sendAsync(req, HttpResponse.BodyHandlers.ofString())
            .thenApply(HttpResponse::body)
            .thenAccept(body -> Platform.runLater(() -> {
                status.setText("Checkout result: " + body);
                cart.clear();
                totalLbl.setText("Total: 0.00");
            }))
            .exceptionally(err -> { Platform.runLater(() -> status.setText("Checkout failed")); return null; });
    }

    private void createCustomer(Label status, String id, String name, String email, String phone) {
        if (id == null || id.isBlank() || name == null || name.isBlank()) { status.setText("Customer ID and Name required"); return; }
        Customer c = new Customer(id, name, email == null || email.isBlank()? null: email, phone == null || phone.isBlank()? null: phone);
        String payload = gson.toJson(c);
    HttpRequest.Builder b = HttpRequest.newBuilder(URI.create("http://localhost:9090/api/customers"))
        .header("Content-Type", "application/json");
    if (authToken != null) b.header("Authorization", "Bearer " + authToken);
    HttpRequest req = b
                .POST(HttpRequest.BodyPublishers.ofString(payload)).build();
        httpClient.sendAsync(req, HttpResponse.BodyHandlers.ofString())
            .thenApply(HttpResponse::body)
            .thenAccept(body -> Platform.runLater(() -> status.setText("Customer created: " + body)))
            .exceptionally(err -> { Platform.runLater(() -> status.setText("Create customer failed")); return null; });
    }

    private void checkHealth(Label status) {
        HttpRequest req = HttpRequest.newBuilder(URI.create("http://localhost:9090/api/health")).GET().build();
        httpClient.sendAsync(req, HttpResponse.BodyHandlers.ofString())
            .thenApply(HttpResponse::body)
            .thenAccept(body -> Platform.runLater(() -> status.setText("Health: " + body)))
            .exceptionally(err -> { Platform.runLater(() -> status.setText("Health check failed")); return null; });
    }

    private void doLogin(Label status, Label userLbl, Button logoutBtn) {
        Stage dialog = new Stage();
        dialog.setTitle("POS Login");
        VBox box = new VBox(8);
        box.setPadding(new Insets(10));
        TextField username = new TextField(); username.setPromptText("Username");
        TextField password = new TextField(); password.setPromptText("Password");
        Button submit = new Button("Login");
        Label msg = new Label("");
        submit.setOnAction(e -> {
            try {
                var payload = gson.toJson(java.util.Map.of("username", username.getText(), "password", password.getText()));
                HttpRequest req = HttpRequest.newBuilder(URI.create("http://localhost:9090/api/auth/login"))
                    .header("Content-Type", "application/json")
                    .POST(HttpRequest.BodyPublishers.ofString(payload)).build();
                httpClient.sendAsync(req, HttpResponse.BodyHandlers.ofString())
                    .thenAccept(resp -> Platform.runLater(() -> {
                        var json = parseJsonObjectOrNull(resp.body());
                        if (json == null) { msg.setText("Login failed"); return; }
                        boolean statusOk = !json.has("status") || (json.get("status").isJsonPrimitive() && "success".equalsIgnoreCase(json.get("status").getAsString()));
                        var tokEl = json.get("token");
                        String tok = (tokEl != null && tokEl.isJsonPrimitive()) ? tokEl.getAsString() : null;
                        if (statusOk && tok != null && !tok.isBlank()) {
                            authToken = tok;
                            authRole = json.has("role") ? json.get("role").getAsString() : null;
                            authUser = username.getText();
                            userLbl.setText("Logged in: " + authUser + (authRole!=null? " ("+authRole+")":""));
                            logoutBtn.setDisable(false);
                            status.setText("Login successful");
                            dialog.close();
                            loadProducts(status);
                        } else {
                            msg.setText("Invalid credentials");
                        }
                    }))
                    .exceptionally(err -> { Platform.runLater(() -> msg.setText("Login error")); return null; });
            } catch (Exception ex) { msg.setText("Login error"); }
        });
        box.getChildren().addAll(new Label("POS Login"), username, password, submit, msg);
        dialog.setScene(new Scene(box, 300, 180));
        dialog.showAndWait();
    }

    private void doLogout(Label status, Label userLbl, Button logoutBtn) {
        if (authToken != null) {
            try {
                HttpRequest req = HttpRequest.newBuilder(URI.create("http://localhost:9090/api/auth/logout"))
                    .header("Authorization", "Bearer " + authToken)
                    .POST(HttpRequest.BodyPublishers.noBody()).build();
                httpClient.sendAsync(req, HttpResponse.BodyHandlers.ofString());
            } catch (Exception ignored) {}
        }
        authToken = null; authRole = null; authUser = null;
        userLbl.setText("");
        logoutBtn.setDisable(true);
        status.setText("Logged out");
    }

    public static void main(String[] args) { launch(); }
}