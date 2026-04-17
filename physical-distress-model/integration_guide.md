# Emergency Detection Service — Integration Guide
### For Spring Boot Backend Developers

---

## What This Service Does

The Python ML microservice takes a list of user messages and returns only the usernames of people whose messages are classified as emergency or distress.

**You send this:**
```json
[
  { "username": "user1", "text": "I feel like giving up." },
  { "username": "user2", "text": "Had a great day!" },
  { "username": "user3", "text": "I don't want to wake up tomorrow." }
]
```

**You get back this:**
```json
["user1", "user3"]
```

Only the at-risk usernames. Nothing else. Your backend then decides what to do with them.

---

## Step 1 — Set Up the Python Service

### Folder structure expected on the server:
```
your-ml-service/
├── app.py                    ← the FastAPI service file
├── bert_emergency_model/     ← unzip bert_emergency_model.zip here
│   ├── config.json
│   ├── pytorch_model.bin
│   ├── tokenizer_config.json
│   ├── vocab.txt
│   └── ...
└── requirements.txt
```

### requirements.txt
```
fastapi==0.110.0
uvicorn==0.29.0
transformers==4.40.0
torch==2.2.0
pydantic==2.0.0
```

### Install and run:
```bash
pip install -r requirements.txt
python app.py
```

Service will start at `http://localhost:8000`

### Verify it is running:
```bash
curl http://localhost:8000/health
```
Expected response:
```json
{ "status": "ok", "model": "bert_emergency_model", "device": "cpu", "threshold": 0.5 }
```

---

## Step 2 — Call From Spring Boot

Add the WebClient dependency to your `pom.xml`:

```xml
<dependency>
    <groupId>org.springframework.boot</groupId>
    <artifactId>spring-boot-starter-webflux</artifactId>
</dependency>
```

### application.properties
```properties
ml.service.url=http://localhost:8000
```

---

### UserMessage DTO
```java
// UserMessage.java
public class UserMessage {
    private String username;
    private String text;

    public UserMessage() {}

    public UserMessage(String username, String text) {
        this.username = username;
        this.text     = text;
    }

    // getters and setters
    public String getUsername() { return username; }
    public void setUsername(String username) { this.username = username; }
    public String getText()     { return text; }
    public void setText(String text) { this.text = text; }
}
```

---

### EmergencyDetectionService
```java
// EmergencyDetectionService.java
import org.springframework.beans.factory.annotation.Value;
import org.springframework.stereotype.Service;
import org.springframework.web.reactive.function.client.WebClient;
import org.springframework.core.ParameterizedTypeReference;
import reactor.core.publisher.Mono;
import java.util.List;

@Service
public class EmergencyDetectionService {

    private final WebClient webClient;

    public EmergencyDetectionService(
            WebClient.Builder builder,
            @Value("${ml.service.url}") String mlServiceUrl) {
        this.webClient = builder.baseUrl(mlServiceUrl).build();
    }

    /**
     * Sends a batch of user messages to the ML service.
     * Returns list of usernames classified as emergency.
     */
    public List<String> getAtRiskUsers(List<UserMessage> messages) {
        return webClient.post()
                .uri("/classify")
                .bodyValue(messages)
                .retrieve()
                .bodyToMono(new ParameterizedTypeReference<List<String>>() {})
                .block();   // use .subscribe() if you want non-blocking
    }

    /**
     * Same as above but returns probabilities for all users.
     * Useful for logging or admin dashboards.
     */
    public Mono<List<Object>> getDetailedResults(List<UserMessage> messages) {
        return webClient.post()
                .uri("/classify/detailed")
                .bodyValue(messages)
                .retrieve()
                .bodyToMono(new ParameterizedTypeReference<List<Object>>() {});
    }
}
```

---

### EmergencyController — Example Usage
```java
// EmergencyController.java
import org.springframework.web.bind.annotation.*;
import java.util.List;

@RestController
@RequestMapping("/api/messages")
public class EmergencyController {

    private final EmergencyDetectionService detectionService;

    public EmergencyController(EmergencyDetectionService detectionService) {
        this.detectionService = detectionService;
    }

    /**
     * Your frontend hits this endpoint with a batch of messages.
     * Spring Boot forwards them to the ML service and returns at-risk usernames.
     *
     * POST /api/messages/analyze
     * Body: [{"username": "user1", "text": "..."}, ...]
     */
    @PostMapping("/analyze")
    public List<String> analyzeMessages(@RequestBody List<UserMessage> messages) {
        List<String> atRiskUsers = detectionService.getAtRiskUsers(messages);

        // TODO: trigger your alert/notification logic here
        // e.g. notificationService.alertModerators(atRiskUsers);

        return atRiskUsers;
    }
}
```

---

## Step 3 — End-to-End Test

With both services running, test the full flow:

```bash
curl -X POST http://localhost:8080/api/messages/analyze \
  -H "Content-Type: application/json" \
  -d '[
    {"username": "alice", "text": "I feel like giving up on everything."},
    {"username": "bob",   "text": "Just finished a great workout!"},
    {"username": "carol", "text": "There was a fire on my street, everyone run!"}
  ]'
```

Expected response:
```json
["alice", "carol"]
```

---

## Step 4 — Docker (Optional but Recommended)

Create a `Dockerfile` next to `app.py`:

```dockerfile
FROM python:3.10-slim

WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY app.py .
COPY bert_emergency_model/ bert_emergency_model/

EXPOSE 8000
CMD ["python", "app.py"]
```

Build and run:
```bash
docker build -t emergency-detection .
docker run -p 8000:8000 emergency-detection
```

In `application.properties` on the Spring Boot side, change:
```properties
ml.service.url=http://emergency-detection:8000
```
if both containers are on the same Docker network.

---

## API Reference

| Endpoint | Method | Description |
|---|---|---|
| `/health` | GET | Check service is up |
| `/classify` | POST | Returns `["username", ...]` of at-risk users only |
| `/classify/detailed` | POST | Returns probability for every user in the batch |

### /classify Request Body
```json
[
  { "username": "string", "text": "string" }
]
```

### /classify Response
```json
["username1", "username2"]
```

### /classify/detailed Response
```json
[
  { "username": "user1", "probability": 0.87, "emergency": true  },
  { "username": "user2", "probability": 0.11, "emergency": false },
  { "username": "user3", "probability": 0.92, "emergency": true  }
]
```

---

## Threshold Tuning

The default threshold is `0.50`. You can change it in `app.py`:

```python
THRESHOLD = 0.50   # default — balanced precision/recall
THRESHOLD = 0.40   # catches more emergencies, more false alarms
THRESHOLD = 0.65   # fewer false alarms, may miss borderline cases
```

For an emergency platform, **lower is safer** — it is better to flag too many than miss one.

---

## Troubleshooting

| Problem | Fix |
|---|---|
| `Connection refused` on port 8000 | Make sure `python app.py` is running |
| `Model not found` error | Check `bert_emergency_model/` folder exists next to `app.py` |
| Slow first request | Normal — model loads on first call. Subsequent calls are fast. |
| CUDA out of memory | Set `DEVICE = torch.device("cpu")` in `app.py` if no GPU on server |
| Spring Boot `WebClient` timeout | Add `.timeout(Duration.ofSeconds(10))` to the WebClient call |

---

*Built with BERT-base-uncased · Fine-tuned on NLP Disaster Tweets · FastAPI 0.110 · Spring Boot 3.x*
