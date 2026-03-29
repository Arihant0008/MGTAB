# MGTAB — Full-Stack Architecture & Validation Report

**Date:** 29 March 2026  
**Project:** MGTAB — Multi-relational Graph-based Twitter Bot Detection  
**Tech Stack:** Node.js + Express | MongoDB | React (Vite) | Python Inference Microservice  

---

## 1. High-Level System Architecture

```
┌──────────────────────────────────────────────────────────────────────────┐
│                           CLIENT (Browser)                              │
│                                                                          │
│   React App (Vite)                                                       │
│   ┌────────────┐  ┌───────────────┐  ┌────────────┐  ┌──────────────┐   │
│   │  Landing   │  │  Bot Checker  │  │ Analytics  │  │   Batch      │   │
│   │  Page      │  │  Page         │  │ Dashboard  │  │   Upload     │   │
│   └────────────┘  └───────┬───────┘  └────────────┘  └──────────────┘   │
│                           │                                              │
└───────────────────────────┼──────────────────────────────────────────────┘
                            │  HTTP / REST (JSON)
                            ▼
┌──────────────────────────────────────────────────────────────────────────┐
│                     NODE.JS + EXPRESS BACKEND                            │
│                                                                          │
│   Middleware Layer                                                        │
│   ┌──────────┐ ┌──────────┐ ┌──────────┐ ┌───────────┐ ┌─────────────┐  │
│   │  CORS    │ │  Rate    │ │  Auth    │ │  Input    │ │  Error      │  │
│   │  Config  │ │  Limiter │ │  (JWT)   │ │  Validator│ │  Handler    │  │
│   └──────────┘ └──────────┘ └──────────┘ └───────────┘ └─────────────┘  │
│                                                                          │
│   Route Layer                                                            │
│   ┌──────────────────┐ ┌──────────────────┐ ┌────────────────────────┐   │
│   │ /api/auth/*      │ │ /api/predict/*   │ │ /api/analytics/*      │   │
│   │ register, login  │ │ single, batch    │ │ models, logs, history │   │
│   └────────┬─────────┘ └────────┬─────────┘ └───────────┬────────────┘  │
│            │                    │                        │               │
│   Controller Layer              │                        │               │
│   ┌────────▼─────────┐ ┌───────▼──────────┐ ┌──────────▼────────────┐   │
│   │ authController   │ │ predictController│ │ analyticsController   │   │
│   └────────┬─────────┘ └───────┬──────────┘ └──────────┬────────────┘   │
│            │                    │                        │               │
│   Service Layer                 │                        │               │
│   ┌────────▼─────────┐ ┌───────▼──────────┐ ┌──────────▼────────────┐   │
│   │ authService      │ │ inferenceService │ │ analyticsService      │   │
│   │ (bcrypt, JWT)    │ │ (calls Python)   │ │ (aggregation queries) │   │
│   └──────────────────┘ └───────┬──────────┘ └───────────────────────┘   │
│                                │                                         │
│                    HTTP call   │  (localhost:5000)                        │
│                                ▼                                         │
│   ┌────────────────────────────────────────────────────────────────────┐  │
│   │         PYTHON INFERENCE MICROSERVICE (Flask/FastAPI)              │  │
│   │                                                                    │  │
│   │   ┌──────────────────┐    ┌────────────────────────────────────┐   │  │
│   │   │  Load best_rgcn  │    │  Load graph_data.pt               │   │  │
│   │   │  .pt at startup  │    │  (features, edges, masks)          │   │  │
│   │   └────────┬─────────┘    └──────────────┬─────────────────────┘   │  │
│   │            │                             │                         │  │
│   │            ▼                             ▼                         │  │
│   │   ┌─────────────────────────────────────────────────────────────┐  │  │
│   │   │               RGCN Model Inference Engine                   │  │  │
│   │   │  Input: node_index → Output: { label, confidence }         │  │  │
│   │   └─────────────────────────────────────────────────────────────┘  │  │
│   └────────────────────────────────────────────────────────────────────┘  │
│                                                                          │
│   Database Layer (Mongoose ODM)                                          │
│   ┌──────────────────────────────────────────────────────────────────┐   │
│   │                        MongoDB Atlas / Local                     │   │
│   │   ┌──────────┐  ┌──────────────┐  ┌──────────────────────────┐  │   │
│   │   │  Users   │  │  Predictions │  │  BatchJobs               │  │   │
│   │   │Collection│  │  Collection  │  │  Collection              │  │   │
│   │   └──────────┘  └──────────────┘  └──────────────────────────┘  │   │
│   └──────────────────────────────────────────────────────────────────┘   │
└──────────────────────────────────────────────────────────────────────────┘
```

---

## 2. Folder Structure (What We Will Build)

```
MGTAB/
├── client/                          # ← REACT FRONTEND (Vite)
│   ├── public/
│   │   └── favicon.ico
│   ├── src/
│   │   ├── assets/                  # images, icons, fonts
│   │   ├── components/
│   │   │   ├── Navbar.jsx
│   │   │   ├── Footer.jsx
│   │   │   ├── HeroSection.jsx
│   │   │   ├── PredictionCard.jsx
│   │   │   ├── ConfidenceMeter.jsx
│   │   │   ├── ModelComparisonChart.jsx
│   │   │   ├── TrainingCurveChart.jsx
│   │   │   ├── ConfusionMatrixHeatmap.jsx
│   │   │   ├── BatchUploader.jsx
│   │   │   ├── HistoryTable.jsx
│   │   │   ├── ProtectedRoute.jsx
│   │   │   └── Loader.jsx
│   │   ├── pages/
│   │   │   ├── LandingPage.jsx
│   │   │   ├── CheckerPage.jsx
│   │   │   ├── DashboardPage.jsx
│   │   │   ├── BatchPage.jsx
│   │   │   ├── LoginPage.jsx
│   │   │   ├── RegisterPage.jsx
│   │   │   └── AboutPage.jsx
│   │   ├── services/
│   │   │   ├── api.js               # Axios instance + interceptors
│   │   │   ├── authService.js
│   │   │   ├── predictService.js
│   │   │   └── analyticsService.js
│   │   ├── context/
│   │   │   └── AuthContext.jsx
│   │   ├── hooks/
│   │   │   ├── useAuth.js
│   │   │   └── usePrediction.js
│   │   ├── utils/
│   │   │   └── validators.js
│   │   ├── App.jsx
│   │   ├── App.css
│   │   ├── index.css
│   │   └── main.jsx
│   ├── index.html
│   ├── package.json
│   └── vite.config.js
│
├── server/                          # ← NODE.JS + EXPRESS BACKEND
│   ├── config/
│   │   ├── db.js                    # MongoDB connection (Mongoose)
│   │   ├── cors.js                  # CORS whitelist
│   │   └── env.js                   # dotenv loader
│   ├── middleware/
│   │   ├── authMiddleware.js        # JWT verification
│   │   ├── rateLimiter.js           # express-rate-limit config
│   │   ├── validator.js             # express-validator rules
│   │   └── errorHandler.js          # Global error handler
│   ├── models/
│   │   ├── User.js                  # Mongoose schema
│   │   ├── Prediction.js            # Mongoose schema
│   │   └── BatchJob.js              # Mongoose schema
│   ├── routes/
│   │   ├── authRoutes.js
│   │   ├── predictRoutes.js
│   │   └── analyticsRoutes.js
│   ├── controllers/
│   │   ├── authController.js
│   │   ├── predictController.js
│   │   └── analyticsController.js
│   ├── services/
│   │   ├── authService.js           # Password hashing, token gen
│   │   ├── inferenceService.js      # HTTP call to Python microservice
│   │   └── analyticsService.js      # DB aggregation queries
│   ├── utils/
│   │   ├── logger.js                # Winston logger
│   │   └── responseHelper.js        # Standardised JSON responses
│   ├── server.js                    # Entry point
│   ├── package.json
│   └── .env
│
├── inference/                       # ← PYTHON MICROSERVICE
│   ├── app.py                       # Flask/FastAPI server
│   ├── model_loader.py              # Loads RGCN + graph_data.pt
│   ├── predict.py                   # Inference logic
│   └── requirements.txt
│
├── Dataset/                         # (existing) raw tensor files
├── 1. Step - Check Shape/           # (existing) data validation
├── ...                              # (existing pipeline steps)
├── best_rgcn.pt                     # (existing) best model weights
├── graph_data.pt                    # (existing) constructed graph
└── PROJECT_STATUS_REPORT.md         # (existing) status report
```

---

## 3. MongoDB Schema Design

### 3.1 Users Collection

```javascript
// models/User.js
{
  _id:          ObjectId,
  name:         { type: String, required: true, trim: true, minlength: 2, maxlength: 50 },
  email:        { type: String, required: true, unique: true, lowercase: true, match: /regex/ },
  password:     { type: String, required: true, minlength: 8 },   // bcrypt hashed
  role:         { type: String, enum: ["user", "admin"], default: "user" },
  totalQueries: { type: Number, default: 0 },
  createdAt:    { type: Date, default: Date.now },
  updatedAt:    { type: Date, default: Date.now }
}
// Indexes: { email: 1 } (unique)
```

### 3.2 Predictions Collection

```javascript
// models/Prediction.js
{
  _id:            ObjectId,
  userId:         { type: ObjectId, ref: "User", required: true },
  inputUsername:   { type: String, required: true, trim: true },
  nodeIndex:      { type: Number, default: null },           // graph node index if found
  prediction:     { type: String, enum: ["bot", "human"], required: true },
  confidence:     { type: Number, min: 0, max: 1, required: true },
  riskLevel:      { type: String, enum: ["low", "medium", "high"], required: true },
  modelUsed:      { type: String, default: "RGCN" },
  inferenceTimeMs:{ type: Number },
  createdAt:      { type: Date, default: Date.now }
}
// Indexes: { userId: 1, createdAt: -1 }, { inputUsername: 1 }
```

### 3.3 BatchJobs Collection

```javascript
// models/BatchJob.js
{
  _id:           ObjectId,
  userId:        { type: ObjectId, ref: "User", required: true },
  status:        { type: String, enum: ["pending", "processing", "completed", "failed"], default: "pending" },
  totalAccounts: { type: Number, required: true },
  processedCount:{ type: Number, default: 0 },
  botsFound:     { type: Number, default: 0 },
  humansFound:   { type: Number, default: 0 },
  results:       [{
    username:    String,
    prediction:  String,
    confidence:  Number,
    riskLevel:   String
  }],
  uploadedAt:    { type: Date, default: Date.now },
  completedAt:   { type: Date, default: null }
}
// Indexes: { userId: 1, status: 1 }
```

---

## 4. Complete API Endpoints

### 4.1 Authentication Routes (`/api/auth`)

| Method | Endpoint | Auth | Description |
|--------|----------|------|-------------|
| `POST` | `/api/auth/register` | ❌ | Create new user account |
| `POST` | `/api/auth/login` | ❌ | Login, returns JWT token |
| `GET` | `/api/auth/me` | ✅ JWT | Get current user profile |
| `PUT` | `/api/auth/update` | ✅ JWT | Update user profile |

### 4.2 Prediction Routes (`/api/predict`)

| Method | Endpoint | Auth | Description |
|--------|----------|------|-------------|
| `POST` | `/api/predict/single` | ✅ JWT | Predict single account |
| `POST` | `/api/predict/batch` | ✅ JWT | Upload CSV for batch prediction |
| `GET` | `/api/predict/history` | ✅ JWT | Get user's prediction history |
| `GET` | `/api/predict/history/:id` | ✅ JWT | Get single prediction detail |
| `DELETE` | `/api/predict/history/:id` | ✅ JWT | Delete a prediction record |

### 4.3 Analytics Routes (`/api/analytics`)

| Method | Endpoint | Auth | Description |
|--------|----------|------|-------------|
| `GET` | `/api/analytics/models` | ❌ | Get model comparison data (all 4 models) |
| `GET` | `/api/analytics/training-logs/:model` | ❌ | Get epoch-wise training CSV data |
| `GET` | `/api/analytics/stats` | ✅ JWT | Get user's personal stats |
| `GET` | `/api/analytics/global-stats` | ❌ | Get global platform stats |

### 4.4 Health Route

| Method | Endpoint | Auth | Description |
|--------|----------|------|-------------|
| `GET` | `/api/health` | ❌ | Backend + Python service health check |

---

## 5. Request/Response Contracts

### 5.1 POST `/api/auth/register`

**Request:**
```json
{
  "name": "Arihant",
  "email": "arihant@example.com",
  "password": "SecurePass123!",
  "confirmPassword": "SecurePass123!"
}
```

**Success Response (201):**
```json
{
  "success": true,
  "message": "Registration successful",
  "data": {
    "token": "eyJhbGciOiJIUzI1NiIs...",
    "user": { "id": "...", "name": "Arihant", "email": "arihant@example.com", "role": "user" }
  }
}
```

**Error Response (400):**
```json
{
  "success": false,
  "message": "Validation failed",
  "errors": [
    { "field": "email", "message": "Email already registered" },
    { "field": "password", "message": "Must be at least 8 characters with 1 uppercase, 1 number, 1 special char" }
  ]
}
```

### 5.2 POST `/api/predict/single`

**Request:**
```json
{
  "username": "suspicious_account_42"
}
```

**Success Response (200):**
```json
{
  "success": true,
  "data": {
    "predictionId": "660f...",
    "username": "suspicious_account_42",
    "prediction": "bot",
    "confidence": 0.9247,
    "riskLevel": "high",
    "modelUsed": "RGCN",
    "inferenceTimeMs": 42,
    "timestamp": "2026-03-29T18:30:00Z"
  }
}
```

### 5.3 POST `/api/predict/batch`

**Request:** `multipart/form-data` with CSV file upload

**Success Response (202):**
```json
{
  "success": true,
  "message": "Batch job started",
  "data": {
    "jobId": "660f...",
    "totalAccounts": 150,
    "status": "processing"
  }
}
```

---

## 6. Complete Validation Rules

### 6.1 Backend Validations (express-validator)

#### Auth Validations

| Field | Rules |
|-------|-------|
| `name` | Required, string, trim, 2–50 chars, no special chars except space |
| `email` | Required, valid email format, normalised to lowercase, unique in DB |
| `password` | Required, min 8 chars, must contain: 1 uppercase, 1 lowercase, 1 digit, 1 special char (`!@#$%^&*`) |
| `confirmPassword` | Required, must exactly match `password` |

#### Predict Validations

| Field | Rules |
|-------|-------|
| `username` | Required, string, trim, 1–50 chars, alphanumeric + underscore only (Twitter format), sanitised against XSS |
| Batch CSV | Max file size: 5MB, must be `.csv`, max 500 rows, each row must have a valid `username` column |

#### Analytics Validations

| Field | Rules |
|-------|-------|
| `:model` param | Must be one of: `gcn`, `gat`, `graphsage`, `rgcn` |
| `page` query | Optional, positive integer, default 1 |
| `limit` query | Optional, 1–100, default 20 |

### 6.2 Frontend Validations (React — real-time)

| Page | Field | Client-Side Rule |
|------|-------|-----------------|
| Register | Name | Non-empty, 2–50 chars, show char counter |
| Register | Email | Valid email regex, debounced uniqueness check via API |
| Register | Password | Min 8 chars, strength meter (weak/medium/strong), show/hide toggle |
| Register | Confirm Password | Must match password (live comparison) |
| Login | Email | Non-empty, valid email format |
| Login | Password | Non-empty |
| Checker | Username | Non-empty, 1–50 chars, alphanumeric + `_` only, no spaces |
| Batch | File | Must be `.csv`, max 5MB, preview first 5 rows before submit |

### 6.3 Security Validations

| Layer | Validation |
|-------|-----------|
| **XSS** | Sanitise all string inputs with `express-validator` `.escape()` + `xss` library |
| **SQL/NoSQL Injection** | Mongoose parameterised queries (built-in), sanitise `$` operators with `express-mongo-sanitize` |
| **Rate Limiting** | `/api/auth/*`: 10 requests/15 min per IP; `/api/predict/*`: 30 requests/min per user |
| **JWT** | Verify token on every protected route, 24h expiry, refresh not needed for MVP |
| **Password** | bcrypt hash with salt rounds = 12, never store or return plaintext |
| **File Upload** | Multer with file size limit (5MB), file type whitelist (`.csv` only), virus scan if scaling |
| **CORS** | Whitelist only `http://localhost:5173` (dev) and production domain |
| **Headers** | Use `helmet` middleware for security headers (CSP, HSTS, X-Frame) |
| **MongoDB** | Connection string in `.env`, never committed; use MongoDB Atlas with IP whitelist in prod |

---

## 7. Frontend Page-by-Page Specification

### 7.1 Landing Page (`/`)

| Section | Content |
|---------|---------|
| **Navbar** | Logo, links (Home, Checker, Dashboard, Batch, About), Login/Register buttons (or avatar if logged in) |
| **Hero** | Large title "Detect Twitter Bots with Graph AI", subtitle, animated CTA button → Checker page |
| **How It Works** | 3-step visual: Input Username → GNN Analyzes Graph → Get Result. Use icons + micro-animations |
| **Model Stats** | 4 cards showing each model's test accuracy + bot recall. Highlight RGCN as champion |
| **Footer** | Credits, GitHub link, tech stack badges |

### 7.2 Checker Page (`/check`)

| Element | Detail |
|---------|--------|
| **Input** | Large search bar with placeholder "Enter Twitter username...", submit button with loading state |
| **Validation** | Real-time: strip `@`, reject spaces/special chars, max 50 chars |
| **Result Card** | Animated reveal: prediction (BOT/HUMAN), confidence percentage (circular gauge), risk badge (Low=green, Medium=yellow, High=red) |
| **History Sidebar** | Recent 10 predictions (auto-populated), click to re-view |
| **Empty State** | Illustration + "Enter a username above to start" |
| **Error State** | "Username not found in dataset" or "Service temporarily unavailable" |

### 7.3 Analytics Dashboard (`/dashboard`)

| Widget | Visualisation |
|--------|--------------|
| **Model Comparison** | Grouped bar chart (Test Accuracy + Bot Recall for all 4 models) — Recharts |
| **Training Curves** | Multi-line chart: Loss, Train Acc, Val Acc over 200 epochs, with model selector dropdown |
| **Confusion Matrix** | Heatmap for RGCN (TP, TN, FP, FN) with color gradient |
| **Dataset Stats** | Pie chart (Bot vs Human distribution), total nodes, total edges |
| **Personal Stats** (auth required) | Total queries made, bots detected, humans detected, last query time |

### 7.4 Batch Page (`/batch`)

| Step | Detail |
|------|--------|
| **Upload** | Drag-and-drop zone or file picker, CSV only, max 5MB |
| **Preview** | Table showing first 5 rows of uploaded CSV |
| **Processing** | Progress bar with live status ("Processing 47/150...") |
| **Results** | Table with columns: Username, Prediction, Confidence, Risk. Sortable + filterable |
| **Export** | Download results as CSV button |

### 7.5 Login / Register Pages (`/login`, `/register`)

| Element | Detail |
|---------|--------|
| **Form** | Clean card layout, centered, with MGTAB branding |
| **Password** | Show/hide toggle, strength meter on register |
| **Error Display** | Inline field-level errors (red text below field) |
| **Redirect** | After login → Checker page; After register → Checker page |
| **Link** | "Don't have an account? Register" / "Already registered? Login" |

### 7.6 About Page (`/about`)

- Project description, MGTAB dataset citation, GNN methodology overview
- Architecture diagram (from this document)
- Team member names + links

---

## 8. Data Flow Diagrams

### 8.1 Single Prediction Flow

```
User enters username on Checker Page
        │
        ▼
[Frontend] validates input (alphanumeric + _, max 50 chars)
        │
        ▼
[Frontend] POST /api/predict/single { username }
        │  (Authorization: Bearer <JWT>)
        ▼
[Express] authMiddleware → verifies JWT
        │
        ▼
[Express] validator middleware → sanitises username
        │
        ▼
[predictController] → calls inferenceService.predict(username)
        │
        ▼
[inferenceService] → HTTP POST to Python service (localhost:5000/predict)
        │  { "username": "..." }
        ▼
[Python Flask] → looks up username in dataset → gets node index
        │      → runs RGCN forward pass → softmax → confidence
        ▼
[Python Flask] → returns { "prediction": "bot", "confidence": 0.9247 }
        │
        ▼
[predictController] → determines riskLevel from confidence
        │   confidence >= 0.8 → "high"
        │   confidence >= 0.5 → "medium"
        │   confidence <  0.5 → "low"
        │
        ▼
[predictController] → saves to MongoDB Predictions collection
        │              → increments user.totalQueries
        ▼
[Express] → returns JSON response to frontend
        │
        ▼
[Frontend] → renders PredictionCard with animated confidence gauge
```

### 8.2 Authentication Flow

```
User fills Register form
        │
        ▼
[Frontend] validates: name (2-50), email (format), password (strength), confirmPassword (match)
        │
        ▼
[Frontend] POST /api/auth/register { name, email, password, confirmPassword }
        │
        ▼
[Express] validator middleware → server-side validation of all fields
        │
        ▼
[authController] → checks if email already exists in Users collection
        │
        ▼
[authService] → bcrypt.hash(password, 12) → save User to MongoDB
        │
        ▼
[authService] → jwt.sign({ userId, role }, SECRET, { expiresIn: "24h" })
        │
        ▼
[Express] → returns { token, user } → 201 Created
        │
        ▼
[Frontend] → stores token in localStorage → updates AuthContext → redirects to /check
```

---

## 9. Environment Variables

```env
# server/.env

# Server
PORT=4000
NODE_ENV=development

# MongoDB
MONGO_URI=mongodb://localhost:27017/mgtab
# or MongoDB Atlas: mongodb+srv://user:pass@cluster.mongodb.net/mgtab

# JWT
JWT_SECRET=your_super_secret_key_here_change_in_prod
JWT_EXPIRES_IN=24h

# Python Inference Service
PYTHON_SERVICE_URL=http://localhost:5000

# Rate Limiting
RATE_LIMIT_WINDOW_MS=900000
RATE_LIMIT_MAX_AUTH=10
RATE_LIMIT_MAX_PREDICT=30

# CORS
CLIENT_URL=http://localhost:5173
```

---

## 10. NPM Dependencies

### Backend (`server/package.json`)

| Package | Purpose |
|---------|---------|
| `express` | Web framework |
| `mongoose` | MongoDB ODM |
| `bcryptjs` | Password hashing |
| `jsonwebtoken` | JWT generation & verification |
| `express-validator` | Input validation & sanitisation |
| `express-rate-limit` | Rate limiting |
| `express-mongo-sanitize` | Prevent NoSQL injection |
| `helmet` | Security headers |
| `cors` | CORS configuration |
| `multer` | File upload (CSV) |
| `csv-parser` | Parse uploaded CSV files |
| `axios` | HTTP client (calls Python service) |
| `winston` | Structured logging |
| `dotenv` | Environment variable loader |
| `morgan` | HTTP request logging |
| `nodemon` (dev) | Auto-restart on changes |

### Frontend (`client/package.json`)

| Package | Purpose |
|---------|---------|
| `react` + `react-dom` | UI library |
| `react-router-dom` | Client-side routing |
| `axios` | HTTP client |
| `recharts` | Charts (bar, line, pie, heatmap) |
| `react-icons` | Icon set |
| `react-hot-toast` | Toast notifications |
| `react-dropzone` | Drag-and-drop file upload |
| `framer-motion` | Animations & transitions |

### Python Inference (`inference/requirements.txt`)

| Package | Purpose |
|---------|---------|
| `flask` or `fastapi` + `uvicorn` | Microservice server |
| `torch` | PyTorch runtime |
| `torch-geometric` | GNN layers |
| `numpy` | Tensor operations |

---

## 11. Communication Between Services

```
┌─────────────┐          ┌──────────────────┐          ┌─────────────────┐
│   React     │  :5173   │   Express API    │  :4000   │  Python Flask   │ :5000
│   Frontend  │ ───────► │   (Node.js)      │ ───────► │  Inference      │
│             │  REST    │                  │  HTTP    │  Microservice   │
│             │  JSON    │                  │  JSON   │                 │
└─────────────┘          └────────┬─────────┘          └─────────────────┘
                                  │
                                  │ Mongoose
                                  ▼
                         ┌────────────────────┐
                         │     MongoDB        │ :27017
                         │  (local or Atlas)  │
                         └────────────────────┘
```

| Service | Port | Responsibility |
|---------|------|---------------|
| React Dev Server (Vite) | `5173` | UI, routing, client validation |
| Express API | `4000` | Auth, CRUD, orchestration, server validation |
| Python Inference | `5000` | Model loading, GNN inference, returns predictions |
| MongoDB | `27017` | Persistent storage (users, predictions, batch jobs) |

---

## 12. Error Handling Strategy

### Standardised Error Response Format

```json
{
  "success": false,
  "message": "Human-readable error message",
  "errors": [
    { "field": "email", "message": "Invalid email format" }
  ],
  "statusCode": 400
}
```

### Error Types & HTTP Codes

| Scenario | Status Code | Message |
|----------|-------------|---------|
| Missing/invalid fields | `400` | "Validation failed" + field-level errors |
| Wrong email/password | `401` | "Invalid credentials" |
| No/expired JWT token | `401` | "Authentication required" |
| Accessing another user's data | `403` | "Access denied" |
| User/Prediction not found | `404` | "Resource not found" |
| Duplicate email registration | `409` | "Email already registered" |
| Rate limit exceeded | `429` | "Too many requests, try again later" |
| Python service down | `503` | "Inference service unavailable" |
| Unexpected server error | `500` | "Internal server error" (logged, details hidden from client) |

---

## 13. Testing & Verification Plan

### 13.1 Backend Tests

| Test Type | Tool | What to Test |
|-----------|------|-------------|
| Unit | `jest` + `supertest` | Each route handler, validation rules, service functions |
| Integration | `jest` + in-memory MongoDB (`mongodb-memory-server`) | Full request → DB flow |
| API | `Postman` or `Thunder Client` | Manual endpoint testing during development |

**Key test cases:**
1. Register with valid data → 201 + token returned
2. Register with duplicate email → 409
3. Register with weak password → 400 with specific error
4. Login with correct creds → 200 + token
5. Login with wrong password → 401
6. Predict without JWT → 401
7. Predict with valid username → 200 + prediction data
8. Predict with invalid username (special chars) → 400
9. Batch upload with valid CSV → 202
10. Batch upload with oversized file → 400
11. Rate limit exceeded → 429
12. Python service down → 503 graceful error

### 13.2 Frontend Tests

| Test Type | Tool | What to Test |
|-----------|------|-------------|
| Component | `Vitest` + `React Testing Library` | Form validation, rendering, state changes |
| E2E | `Browser subagent` / `Cypress` | Full user flows (register → login → predict → view history) |

### 13.3 Python Inference Tests

| Test | Method |
|------|--------|
| Model loads without error | Run `app.py`, check startup logs |
| `/predict` returns valid JSON | `curl` / `httpie` with test node index |
| Invalid input returns error | Send empty body, check 400 response |

---

## 14. Deployment Roadmap (Future)

| Stage | Action |
|-------|--------|
| **Dev** | All 3 services running locally (ports 5173, 4000, 5000) |
| **Staging** | Docker Compose with 3 containers + MongoDB container |
| **Prod** | Frontend → Vercel/Netlify, Express → Railway/Render, Python → Render, MongoDB → Atlas |

---

## 15. Summary: Build Order

```
Phase 1 (Days 1–3): Foundation
  ├── Set up server/ with Express boilerplate
  ├── Set up MongoDB connection + schemas
  ├── Build auth routes (register/login) + JWT
  └── Set up inference/ Python microservice

Phase 2 (Days 4–6): Core Features
  ├── Build /predict/single endpoint + Python integration
  ├── Build /predict/history endpoint
  └── Set up client/ with Vite + React + routing

Phase 3 (Days 7–10): Frontend UI
  ├── Landing page with hero + model stats
  ├── Checker page with input + result card
  ├── Login / Register pages
  └── Connect all pages to backend API

Phase 4 (Days 11–14): Advanced Features
  ├── Analytics dashboard with charts
  ├── Batch upload page
  ├── History page with search/filter
  └── Polish animations + responsive design

Phase 5 (Days 15–17): Hardening
  ├── Rate limiting, helmet, error handling
  ├── Testing (backend + frontend)
  └── Documentation / About page
```

---

*This architecture report serves as the complete technical blueprint for building the MGTAB full-stack application. All validations, schemas, endpoints, and data flows are specified above. No code changes have been made — this is a planning document only.*
