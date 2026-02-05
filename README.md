# LLM-Gateway Implementierungskonzept v1.3 

**Kostenoptimiertes AI-Routing für OpenClaw**

Version 1.3 (Cost-Optimized) – Februar 2026
Mit Vibe Code Beispiel 

> **Hetzner + Groq Router + Prompt Caching = 73% Kostenreduktion**
> 
> Monatliche Kosten: ~€25-30 statt €92 (AWS-Baseline)

---

## Inhaltsverzeichnis

1. [Executive Summary](#1-executive-summary)
2. [Architektur-Übersicht](#2-architektur-übersicht)
3. [Infrastruktur: Hetzner statt AWS](#3-infrastruktur-hetzner-statt-aws)
4. [Dreistufiges Routing](#4-dreistufiges-routing)
5. [Groq als Router](#5-groq-als-router)
6. [Hard Policy Gate (Sicherheit)](#6-hard-policy-gate-sicherheit)
7. [Rate Limiting & Kill Switch](#7-rate-limiting--kill-switch)
8. [Zweistufiges Caching](#8-zweistufiges-caching)
9. [Hybrid Retrieval (BM25 + Embeddings)](#9-hybrid-retrieval-bm25--embeddings)
10. [Anthropic Prompt Caching](#10-anthropic-prompt-caching)
11. [Context Budgeting & Compression](#11-context-budgeting--compression)
12. [Capability-based Tools](#12-capability-based-tools)
13. [Deterministisches Patching](#13-deterministisches-patching)
14. [Patch Risk Score](#14-patch-risk-score)
15. [Risk-Stratified Verifier](#15-risk-stratified-verifier)
16. [Monitoring & KPIs](#16-monitoring--kpis)
17. [Implementierungsplan](#17-implementierungsplan)
18. [Kostenprognose](#18-kostenprognose)
19. [Changelog](#19-changelog)

---

## 1. Executive Summary

### 1.1 Problem

OpenClaw (AI Coding Assistant) benötigt LLM-Zugriff für:
- Code-Erklärungen und Dokumentation
- Bug-Fixes und Refactoring-Vorschläge
- Shell-Kommando-Generierung
- Code-Reviews

**Herausforderung:** Claude Sonnet kostet $3/1M Input + $15/1M Output. Bei 100+ Requests/Tag entstehen schnell $100+/Monat.

### 1.2 Lösung

Ein intelligentes Gateway mit:

| Komponente | Funktion | Kosteneffekt |
|------------|----------|--------------|
| **Dreistufiges Routing** | Einfache Fragen → günstige Modelle | -60% API-Kosten |
| **Zweistufiges Caching** | Exact + Semantic Cache | -30% redundante Calls |
| **Groq Router** | Schnelle Intent-Classification | 3x schneller als Ollama |
| **Prompt Caching** | Anthropic cached System-Prompts | -90% System-Prompt-Kosten |
| **Context Budgeting** | Begrenzte Input-Tokens pro Tier | -40% Premium Input |
| **Hetzner Infrastruktur** | Festpreis statt AWS Pay-per-Use | -€60/Monat |

### 1.3 Kostenvergleich

| Szenario | AWS (v1.2) | Hetzner (v1.3) | Ersparnis |
|----------|------------|----------------|-----------|
| Server | $74 (t3.medium) | €8,50 (CX22) | -€60 |
| Speicher | $5 (EBS) | €0 (inkl.) | -€5 |
| Traffic | $5-10 | €0 (20TB inkl.) | -€7 |
| Router | $0 (Ollama) | €1-2 (Groq) | +€1,50 |
| Premium (mit Caching) | $25-35 | €8-12 | -€20 |
| Embeddings + Verifier | $3-5 | €1-2 | -€3 |
| **TOTAL** | **~€92/Monat** | **~€25/Monat** | **-73%** |

### 1.4 Architektur-Entscheidung

| Option | Wann wählen? | Kosten |
|--------|--------------|--------|
| **Hetzner + Groq** | Standard-Empfehlung; niedrige Kosten, einfache Wartung | €25-30/Monat |
| **Hetzner + Ollama** | Wenn 100% lokal/offline nötig; 8GB Server | €35/Monat |
| **AWS + Ollama** | Nur wenn AWS-Ökosystem Pflicht (IAM, VPC) | €90+/Monat |

---

## 2. Architektur-Übersicht

### 2.1 Request-Flow

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                              REQUEST FLOW                                    │
└─────────────────────────────────────────────────────────────────────────────┘

  OpenClaw Request
        │
        ▼
┌───────────────────┐
│  Hard Policy Gate │ ──── BLOCK ───► 403 Forbidden
│  (Sicherheit)     │                 (rm -rf, secrets, etc.)
└───────────────────┘
        │ PASS
        ▼
┌───────────────────┐
│  Rate Limiter     │ ──── BLOCK ───► 429 Too Many Requests
│  + Budget Guard   │                 (Daily Cap erreicht)
└───────────────────┘
        │ PASS
        ▼
┌───────────────────┐
│  Exact Cache      │ ──── HIT ────► Cached Response
│  (SHA-256 Hash)   │
└───────────────────┘
        │ MISS
        ▼
┌───────────────────┐
│  Semantic Cache   │ ──── HIT ────► Verifier ──► Response
│  (Embeddings)     │                    │
└───────────────────┘                    └── INVALID ──► Regenerate
        │ MISS
        ▼
┌───────────────────┐
│  Groq Router      │
│  (Intent-Class.)  │
└───────────────────┘
        │
        ├── CACHE_ONLY ──► "Bitte präzisieren"
        │
        ├── LOCAL ──────► Haiku (günstig)
        │
        ├── CHEAP ──────► Haiku (günstig)
        │
        └── PREMIUM ────► Sonnet (mit Prompt Caching)
                              │
                              ▼
                    ┌─────────────────┐
                    │ Context Budget  │
                    │ + Compression   │
                    └─────────────────┘
                              │
                              ▼
                         Response
                              │
                              ▼
                    ┌─────────────────┐
                    │  Cache Store    │
                    │  + Fingerprint  │
                    └─────────────────┘
```

### 2.2 Komponenten-Übersicht

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                           HETZNER CX22 (4GB RAM)                            │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐        │
│  │   FastAPI   │  │   SQLite    │  │   SQLite    │  │   Nginx     │        │
│  │   Gateway   │  │ Exact Cache │  │Semantic Cache│ │   Reverse   │        │
│  │   (uvicorn) │  │   (FTS5)    │  │ + Embeddings │  │   Proxy     │        │
│  └─────────────┘  └─────────────┘  └─────────────┘  └─────────────┘        │
│                                                                             │
├─────────────────────────────────────────────────────────────────────────────┤
│                           EXTERNE SERVICES                                  │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐        │
│  │    Groq     │  │  Anthropic  │  │   OpenAI    │  │  OpenClaw   │        │
│  │   Router    │  │   Claude    │  │  Embeddings │  │   Client    │        │
│  │  (Llama 8B) │  │   Sonnet    │  │ (Fallback)  │  │             │        │
│  └─────────────┘  └─────────────┘  └─────────────┘  └─────────────┘        │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

---

## 3. Infrastruktur: Hetzner statt AWS

### 3.1 Warum Hetzner?

| Aspekt | AWS t3.medium | Hetzner CX22 | Vorteil |
|--------|---------------|--------------|---------|
| **Preis** | $74/Monat + Extras | €4,35/Monat fix | -90% Basis |
| **RAM** | 4GB (burstable) | 4GB (dediziert) | Kein Throttling |
| **Storage** | $0,10/GB (EBS) | 40GB inkl. | Keine IOPS-Kosten |
| **Traffic** | $0,09/GB out | 20TB inkl. | Keine Überraschungen |
| **Standort** | us-east-1 | Nürnberg/Falkenstein | DSGVO, niedrige Latenz |
| **Komplexität** | IAM, VPC, SG | SSH + Firewall | Einfacher |

### 3.2 Server-Optionen

| Server | RAM | vCPU | SSD | Preis | Use-Case |
|--------|-----|------|-----|-------|----------|
| **CX22** | 4GB | 2 | 40GB | €4,35 | Groq-Router (empfohlen) |
| **CX32** | 8GB | 4 | 80GB | €7,05 | Ollama lokal |
| **CX42** | 16GB | 8 | 160GB | €14,76 | Heavy Workload |

### 3.3 Server-Setup

```bash
# 1. Hetzner Cloud Console: CX22 mit Ubuntu 24.04 LTS erstellen

# 2. SSH-Zugang einrichten
ssh root@<server-ip>

# 3. Basis-Setup
apt update && apt upgrade -y
apt install -y python3-pip python3-venv nginx certbot python3-certbot-nginx

# 4. Firewall konfigurieren
ufw allow 22/tcp
ufw allow 80/tcp
ufw allow 443/tcp
ufw enable

# 5. Non-root User erstellen
adduser gateway
usermod -aG sudo gateway
su - gateway

# 6. Gateway installieren
cd /opt
sudo mkdir llm-gateway && sudo chown gateway:gateway llm-gateway
cd llm-gateway
python3 -m venv venv
source venv/bin/activate

# 7. Dependencies installieren
pip install fastapi uvicorn httpx anthropic openai tenacity sqlite-utils

# 8. Environment Variables
cat > .env << 'EOF'
GROQ_API_KEY=gsk_...
ANTHROPIC_API_KEY=sk-ant-...
OPENAI_API_KEY=sk-...  # Nur für Embeddings-Fallback
GATEWAY_SECRET=<random-secret>
DAILY_BUDGET_HARD=50.0
DAILY_BUDGET_SOFT=5.0
EOF

# 9. Systemd Service
sudo cat > /etc/systemd/system/llm-gateway.service << 'EOF'
[Unit]
Description=LLM Gateway
After=network.target

[Service]
Type=simple
User=gateway
WorkingDirectory=/opt/llm-gateway
Environment="PATH=/opt/llm-gateway/venv/bin"
EnvironmentFile=/opt/llm-gateway/.env
ExecStart=/opt/llm-gateway/venv/bin/uvicorn main:app --host 127.0.0.1 --port 8000
Restart=always
RestartSec=5

[Install]
WantedBy=multi-user.target
EOF

sudo systemctl daemon-reload
sudo systemctl enable llm-gateway
sudo systemctl start llm-gateway

# 10. Nginx Reverse Proxy + SSL
sudo cat > /etc/nginx/sites-available/llm-gateway << 'EOF'
server {
    listen 80;
    server_name gateway.yourdomain.com;
    
    location / {
        proxy_pass http://127.0.0.1:8000;
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
        proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
        proxy_read_timeout 120s;
    }
}
EOF

sudo ln -s /etc/nginx/sites-available/llm-gateway /etc/nginx/sites-enabled/
sudo nginx -t && sudo systemctl reload nginx

# 11. SSL mit Let's Encrypt
sudo certbot --nginx -d gateway.yourdomain.com
```

---

## 4. Dreistufiges Routing

### 4.1 Tier-Definitionen

| Tier | Modell | Kosten | Latenz | Use-Case |
|------|--------|--------|--------|----------|
| **LOCAL** | Haiku (via Groq Fallback) | $0.25/1M | 200ms | Einfache Fragen |
| **CHEAP** | Claude Haiku | $0.25/1M in, $1.25/1M out | 300ms | Erklärungen, Docs |
| **PREMIUM** | Claude Sonnet | $3/1M in, $15/1M out | 500ms | Code-Generierung, Patches |

### 4.2 Router-Logik

```python
from enum import Enum
from pydantic import BaseModel

class RouterAction(str, Enum):
    CACHE_ONLY = "cache_only"    # Zu vage, kein LLM-Call
    LOCAL = "local"              # Einfache Fragen
    CHEAP = "cheap"              # Erklärungen, Dokumentation
    PREMIUM = "premium"          # Code-Generierung, komplexe Tasks

class RouterResult(BaseModel):
    action: RouterAction
    confidence: float           # 0.0 - 1.0
    response_type: str          # Für Cache-TTL und Verifier
    reason: str                 # Für Logging/Debugging

# Response Types für Cache-Management
RESPONSE_TYPES = {
    "explanation_generic": {
        "description": "Allgemeine Erklärungen (Was ist X?)",
        "ttl_base": 7 * 24 * 3600,  # 7 Tage
        "tier": "cheap"
    },
    "explanation_contextual": {
        "description": "Projekt-spezifische Erklärungen",
        "ttl_base": 24 * 3600,  # 24 Stunden
        "tier": "cheap"
    },
    "code_suggestion": {
        "description": "Code-Vorschläge, Patches",
        "ttl_base": 0,  # Invalidiert bei Git-Commit
        "tier": "premium"
    },
    "code_review": {
        "description": "Code-Reviews, Best Practices",
        "ttl_base": 12 * 3600,  # 12 Stunden
        "tier": "premium"
    },
    "command_execution": {
        "description": "Shell-Kommandos, CLI",
        "ttl_base": 3600,  # 1 Stunde
        "tier": "premium"
    },
    "documentation": {
        "description": "API-Docs, README-Generierung",
        "ttl_base": 24 * 3600,
        "tier": "cheap"
    }
}
```

### 4.3 Router System-Prompt

```python
ROUTER_SYSTEM_PROMPT = """Du bist ein Intent-Classifier für Coding-Anfragen.

Klassifiziere die Anfrage in GENAU EINE Kategorie:

CACHE_ONLY - Anfrage ist zu vage/unklar für sinnvolle Antwort
  Beispiele: "help", "?", "code", "fix it"

LOCAL - Triviale Fragen, die keine Code-Analyse brauchen
  Beispiele: "Was bedeutet HTTP 404?", "Wie heißt der Befehl für..."

CHEAP - Erklärungen, Dokumentation, allgemeine Best Practices
  Beispiele: "Erkläre mir async/await", "Was ist der Unterschied zwischen..."

PREMIUM - Code-Generierung, Patches, komplexe Analyse, projektspezifisch
  Beispiele: "Schreibe eine Funktion die...", "Fixe den Bug in...", "Refactore..."

Antworte NUR mit JSON:
{"action": "...", "confidence": 0.0-1.0, "response_type": "...", "reason": "..."}

response_type muss einer von sein:
- explanation_generic
- explanation_contextual  
- code_suggestion
- code_review
- command_execution
- documentation"""
```

---

## 5. Groq als Router

### 5.1 Warum Groq statt Ollama?

| Aspekt | Ollama (lokal) | Groq (API) |
|--------|----------------|------------|
| **Latenz** | 500-800ms (warm), 2-4s (kalt) | 150-250ms (konstant) |
| **RAM-Bedarf** | 2,5 GB (30% von 8GB) | 0 GB |
| **OOM-Risiko** | Ja (mit Cache + Embeddings) | Nein |
| **Wartung** | Updates, Monitoring, Tuning | Keine |
| **Kosten** | €0 (aber größerer Server nötig) | ~€1-2/Monat |
| **Offline-Fähig** | Ja | Nein |

### 5.2 Groq Implementation

```python
import httpx
import os
from tenacity import retry, stop_after_attempt, wait_exponential

GROQ_API_KEY = os.environ["GROQ_API_KEY"]
GROQ_MODEL = "llama-3.1-8b-instant"  # $0.05/1M input, $0.08/1M output

@retry(
    stop=stop_after_attempt(3),
    wait=wait_exponential(multiplier=0.5, min=0.1, max=2)
)
async def groq_classify(query: str, context: str = "") -> RouterResult:
    """Klassifiziert Intent via Groq Llama 3.1 8B"""
    
    user_content = f"Kontext: {context}\n\nAnfrage: {query}" if context else query
    
    async with httpx.AsyncClient() as client:
        response = await client.post(
            "https://api.groq.com/openai/v1/chat/completions",
            headers={
                "Authorization": f"Bearer {GROQ_API_KEY}",
                "Content-Type": "application/json"
            },
            json={
                "model": GROQ_MODEL,
                "messages": [
                    {"role": "system", "content": ROUTER_SYSTEM_PROMPT},
                    {"role": "user", "content": user_content}
                ],
                "max_tokens": 150,
                "temperature": 0,
                "response_format": {"type": "json_object"}
            },
            timeout=5.0
        )
        response.raise_for_status()
        
        result = response.json()["choices"][0]["message"]["content"]
        parsed = json.loads(result)
        
        return RouterResult(
            action=RouterAction(parsed["action"]),
            confidence=float(parsed.get("confidence", 0.8)),
            response_type=parsed.get("response_type", "explanation_generic"),
            reason=parsed.get("reason", "")
        )
```

### 5.3 Kosten-Rechnung Groq

```python
# Annahmen:
# - 500 Requests/Tag
# - 300 Tokens pro Router-Call (System-Prompt + Query + Response)

monthly_requests = 500 * 30  # = 15.000
tokens_per_request = 300
monthly_tokens = monthly_requests * tokens_per_request  # = 4.500.000

# Groq Llama 3.1 8B Instant Pricing (Stand Feb 2026)
input_price = 0.05 / 1_000_000   # $0.05 pro 1M Input
output_price = 0.08 / 1_000_000  # $0.08 pro 1M Output

# ~90% sind Input (System-Prompt + Query), ~10% Output
input_tokens = monthly_tokens * 0.9  # = 4.050.000
output_tokens = monthly_tokens * 0.1  # = 450.000

monthly_cost = (input_tokens * input_price) + (output_tokens * output_price)
# = $0.2025 + $0.036 = $0.24/Monat

# Mit Sicherheitspuffer: ~€1-2/Monat
```

### 5.4 Fallback-Kette

```python
async def route_with_resilience(query: str, context: str = "") -> RouterResult:
    """Router mit Fallback-Kette: Groq → Haiku → Default"""
    
    # Primär: Groq (schnell, günstig)
    try:
        return await groq_classify(query, context)
    except (httpx.TimeoutException, httpx.HTTPStatusError) as e:
        log.warning(f"Groq failed: {e}, falling back to Haiku")
        metrics.increment("router_fallback", tags={"from": "groq", "to": "haiku"})
    
    # Sekundär: Haiku (teurer, aber zuverlässig)
    try:
        return await haiku_classify(query, context)
    except Exception as e:
        log.error(f"Haiku classifier failed: {e}, using default")
        metrics.increment("router_fallback", tags={"from": "haiku", "to": "default"})
    
    # Ultima Ratio: Sicherheitsmodus → Premium
    return RouterResult(
        action=RouterAction.PREMIUM,
        confidence=0.5,
        response_type="code_suggestion",
        reason="fallback_default"
    )
```

---

## 6. Hard Policy Gate (Sicherheit)

### 6.1 Warum VOR dem Router?

Das Hard Policy Gate läuft **VOR** dem LLM-Router, um:
1. Gefährliche Anfragen sofort zu blocken (kein LLM-Call nötig)
2. Prompt-Injection-Versuche abzufangen
3. Kosten für offensichtlich illegitime Anfragen zu sparen

### 6.2 Blocklisten

```python
import re
from typing import Optional

class PolicyViolation(Exception):
    def __init__(self, category: str, pattern: str, message: str):
        self.category = category
        self.pattern = pattern
        self.message = message

# Gefährliche Shell-Kommandos
DANGEROUS_COMMANDS = [
    # Destruktive Operationen
    r"\brm\s+(-[rf]+\s+)*(/|~|\$HOME|\*)",
    r"\bmkfs\b",
    r"\bdd\s+.*of=/dev/",
    r"\b:(){.*};\s*:",  # Fork Bomb
    r"\bchmod\s+(-R\s+)?[0-7]*777",
    r"\bchown\s+-R\s+.*\s+/",
    
    # Netzwerk-Angriffe
    r"\bnc\s+.*-e\s+/bin/(ba)?sh",
    r"\bcurl\s+.*\|\s*(ba)?sh",
    r"\bwget\s+.*-O\s*-\s*\|\s*(ba)?sh",
    
    # Crypto-Mining
    r"\b(xmrig|minerd|cgminer|bfgminer)\b",
    
    # Service-Manipulation
    r"\bsystemctl\s+(stop|disable|mask)\s+(ssh|sshd|ufw|iptables)",
    r"\bservice\s+\w+\s+(stop|disable)",
    
    # Indirekte destruktive Ops (NEU in v1.2)
    r"\bfind\s+.*-delete",
    r"\btruncate\s+--size\s*0",
]

# Sensitive Daten-Patterns
SENSITIVE_PATTERNS = [
    r"(api[_-]?key|secret|password|token)\s*[=:]\s*['\"]?[\w-]{20,}",
    r"-----BEGIN\s+(RSA\s+)?PRIVATE\s+KEY-----",
    r"aws_secret_access_key\s*=",
    r"ghp_[a-zA-Z0-9]{36}",  # GitHub Token
    r"sk-[a-zA-Z0-9]{48}",   # OpenAI Key
    r"sk-ant-[a-zA-Z0-9-]{95}",  # Anthropic Key
]

# Forbidden Paths
FORBIDDEN_PATHS = [
    r"/etc/(passwd|shadow|sudoers)",
    r"/root/",
    r"~/.ssh/",
    r"\.env",
    r"\.(pem|key|crt)$",
]

# Unicode-Normalisierung (gegen Bypass-Versuche)
def normalize_unicode(text: str) -> str:
    """Normalisiert Unicode-Tricks wie unsichtbare Zeichen"""
    import unicodedata
    # NFKC normalisiert ① → 1, ᴿ → R, etc.
    text = unicodedata.normalize("NFKC", text)
    # Entferne Zero-Width-Characters
    text = re.sub(r"[\u200b-\u200f\u2060\ufeff]", "", text)
    return text

def check_hard_policy(query: str) -> Optional[PolicyViolation]:
    """
    Prüft Query gegen Hard Policy Rules.
    Returns None wenn OK, sonst PolicyViolation.
    """
    # Normalisiere zuerst
    normalized = normalize_unicode(query.lower())
    
    # Check dangerous commands
    for pattern in DANGEROUS_COMMANDS:
        if re.search(pattern, normalized, re.IGNORECASE):
            return PolicyViolation(
                category="dangerous_command",
                pattern=pattern,
                message="Gefährliches Kommando erkannt"
            )
    
    # Check sensitive data (auch in Original, nicht nur normalized)
    for pattern in SENSITIVE_PATTERNS:
        if re.search(pattern, query, re.IGNORECASE):
            return PolicyViolation(
                category="sensitive_data",
                pattern=pattern,
                message="Sensible Daten in Anfrage erkannt"
            )
    
    # Check forbidden paths
    for pattern in FORBIDDEN_PATHS:
        if re.search(pattern, normalized, re.IGNORECASE):
            return PolicyViolation(
                category="forbidden_path",
                pattern=pattern,
                message="Zugriff auf geschützten Pfad"
            )
    
    return None
```

### 6.3 Command Allowlist (Capability-Based)

```python
# Statt Blocklist: Explizite Allowlist für Tool-Operationen
ALLOWED_COMMANDS = {
    # Git (read-mostly)
    "git": ["status", "diff", "log", "show", "branch", "remote", "fetch"],
    
    # Package Managers (read + install)
    "npm": ["list", "outdated", "audit", "install", "ci", "run", "test"],
    "pip": ["list", "show", "check", "install"],
    
    # Build Tools
    "make": ["*"],  # Makefile-defined targets
    "cargo": ["build", "test", "check", "clippy", "fmt"],
    
    # Docker (read-mostly)
    "docker": ["ps", "images", "logs", "inspect", "stats"],
    "docker-compose": ["ps", "logs", "config"],
    
    # System Info (read-only)
    "ls": ["*"],
    "cat": ["*"],  # Mit Path-Restriction!
    "head": ["*"],
    "tail": ["*"],
    "grep": ["*"],
    "find": ["-name", "-type", "-mtime"],  # Kein -delete!
    "du": ["*"],
    "df": ["*"],
    "free": ["*"],
    "top": ["-bn1"],
    "ps": ["aux", "-ef"],
}

def is_command_allowed(command: str, args: list[str]) -> bool:
    """Prüft ob Kommando auf Allowlist ist"""
    if command not in ALLOWED_COMMANDS:
        return False
    
    allowed_args = ALLOWED_COMMANDS[command]
    if "*" in allowed_args:
        return True
    
    # Prüfe jedes Argument
    for arg in args:
        if arg.startswith("-"):
            if arg not in allowed_args:
                return False
    
    return True
```

---

## 7. Rate Limiting & Kill Switch

### 7.1 Token-Aware Rate Limiting

```python
from datetime import datetime, timedelta
from collections import defaultdict
import asyncio

class RateLimiter:
    def __init__(self):
        self.limits = {
            "local": {
                "rpm": 100,      # Requests per Minute
                "tpm": 50_000,   # Tokens per Minute
                "daily_usd": 0,  # Unlimited (local)
                "burst": 10      # Burst Buffer
            },
            "cheap": {
                "rpm": 50,
                "tpm": 100_000,
                "daily_usd": 10,
                "burst": 5
            },
            "premium": {
                "rpm": 20,
                "tpm": 50_000,
                "daily_usd": 50,
                "burst": 3
            }
        }
        
        # Tracking per Tier
        self.minute_requests = defaultdict(list)  # tier -> [timestamps]
        self.minute_tokens = defaultdict(int)
        self.daily_spend = defaultdict(float)
        self.last_reset = datetime.now()
    
    def check(self, tier: str, estimated_tokens: int) -> tuple[bool, str, float]:
        """
        Returns: (allowed, reason, delay_seconds)
        """
        now = datetime.now()
        limits = self.limits[tier]
        
        # Daily Reset
        if now.date() != self.last_reset.date():
            self.daily_spend.clear()
            self.last_reset = now
        
        # Cleanup alte Timestamps (>1 Minute)
        cutoff = now - timedelta(minutes=1)
        self.minute_requests[tier] = [
            ts for ts in self.minute_requests[tier] if ts > cutoff
        ]
        
        # RPM Check
        if len(self.minute_requests[tier]) >= limits["rpm"]:
            # Burst erlauben?
            if len(self.minute_requests[tier]) < limits["rpm"] + limits["burst"]:
                return (True, "burst_allowed", 0)
            return (False, "rpm_exceeded", 60)
        
        # TPM Check
        if self.minute_tokens[tier] + estimated_tokens > limits["tpm"]:
            return (False, "tpm_exceeded", 30)
        
        # Daily Budget Check (Premium)
        if limits["daily_usd"] > 0:
            estimated_cost = self._estimate_cost(tier, estimated_tokens)
            if self.daily_spend[tier] + estimated_cost > limits["daily_usd"]:
                return (False, "daily_budget_exceeded", 0)
        
        return (True, "ok", 0)
    
    def record(self, tier: str, tokens_used: int, cost_usd: float):
        """Zeichnet erfolgreichen Request auf"""
        self.minute_requests[tier].append(datetime.now())
        self.minute_tokens[tier] += tokens_used
        self.daily_spend[tier] += cost_usd
    
    def _estimate_cost(self, tier: str, tokens: int) -> float:
        """Schätzt Kosten basierend auf Tier"""
        prices = {
            "local": 0,
            "cheap": 0.25 / 1_000_000,   # Haiku Input
            "premium": 3.00 / 1_000_000   # Sonnet Input (Output separat)
        }
        return tokens * prices.get(tier, 0)
```

### 7.2 Global Kill Switch (NEU in v1.3)

```python
class BudgetGuard:
    """
    Dreistufiges Budget-System:
    - Soft: Warning + Log
    - Medium: Throttle (5s Delay)
    - Hard: Kill Switch (Premium disabled)
    """
    
    def __init__(self):
        self.limits = {
            "soft": float(os.environ.get("DAILY_BUDGET_SOFT", "5.0")),
            "medium": float(os.environ.get("DAILY_BUDGET_MEDIUM", "15.0")),
            "hard": float(os.environ.get("DAILY_BUDGET_HARD", "50.0"))
        }
        self.today_spend = 0.0
        self.premium_disabled = False
        self.last_reset = datetime.now().date()
    
    def check(self, tier: str, estimated_cost: float) -> dict:
        """
        Returns: {
            "allowed": bool,
            "delay": float,
            "reason": str
        }
        """
        # Daily Reset
        today = datetime.now().date()
        if today != self.last_reset:
            self.today_spend = 0.0
            self.premium_disabled = False
            self.last_reset = today
            log.info("Daily budget reset")
        
        projected = self.today_spend + estimated_cost
        
        # Kill Switch aktiv?
        if self.premium_disabled:
            if tier == "premium":
                return {
                    "allowed": False,
                    "delay": 0,
                    "reason": "kill_switch_active"
                }
            # Local/Cheap noch erlaubt
            return {"allowed": True, "delay": 0, "reason": "non_premium_allowed"}
        
        # Hard Limit → Kill Switch aktivieren
        if projected > self.limits["hard"]:
            self.premium_disabled = True
            log.critical(f"KILL SWITCH ACTIVATED: ${projected:.2f} > ${self.limits['hard']}")
            self._alert_admin(f"Budget Kill Switch: ${projected:.2f}")
            return {
                "allowed": False,
                "delay": 0,
                "reason": "hard_limit_kill_switch"
            }
        
        # Medium Limit → Throttle
        if projected > self.limits["medium"]:
            log.error(f"Budget ${projected:.2f} > Medium ${self.limits['medium']}, THROTTLING")
            return {
                "allowed": True,
                "delay": 5.0,
                "reason": "throttle_medium_limit"
            }
        
        # Soft Limit → Warning
        if projected > self.limits["soft"]:
            log.warning(f"Budget ${projected:.2f} > Soft ${self.limits['soft']}")
        
        return {"allowed": True, "delay": 0, "reason": "ok"}
    
    def record_spend(self, cost: float):
        """Zeichnet tatsächliche Kosten auf"""
        self.today_spend += cost
        metrics.gauge("daily_spend_usd", self.today_spend)
    
    def _alert_admin(self, message: str):
        """Sendet Alert (Webhook, E-Mail, etc.)"""
        # TODO: Implement alerting
        pass
```

### 7.3 Idempotency Keys (Verhindert Doppel-Calls)

```python
import hashlib
from datetime import datetime, timedelta

class IdempotencyGuard:
    """
    Verhindert doppelte API-Calls bei Retries/Timeouts.
    Cached Response für identische Requests für 5 Minuten.
    """
    
    def __init__(self, ttl_seconds: int = 300):
        self.cache: dict[str, tuple[dict, datetime]] = {}
        self.ttl = timedelta(seconds=ttl_seconds)
    
    def get_key(self, request: dict) -> str:
        """Generiert eindeutigen Key aus Request-Inhalt"""
        content = json.dumps({
            "model": request.get("model"),
            "messages": request.get("messages"),
            "temperature": request.get("temperature", 1),
            "max_tokens": request.get("max_tokens")
        }, sort_keys=True)
        return hashlib.sha256(content.encode()).hexdigest()[:16]
    
    def check(self, key: str) -> Optional[dict]:
        """Prüft ob Request bereits verarbeitet wurde"""
        self._cleanup()
        
        if key in self.cache:
            response, timestamp = self.cache[key]
            log.info(f"Idempotency hit: {key}")
            metrics.increment("idempotency_hits")
            return response
        
        return None
    
    def store(self, key: str, response: dict):
        """Speichert Response für Key"""
        self.cache[key] = (response, datetime.now())
    
    def _cleanup(self):
        """Entfernt abgelaufene Einträge"""
        now = datetime.now()
        expired = [k for k, (_, ts) in self.cache.items() if now - ts > self.ttl]
        for k in expired:
            del self.cache[k]
```

---

## 8. Zweistufiges Caching

### 8.1 Cache-Architektur

```
┌─────────────────────────────────────────────────────────────────────┐
│                         CACHE FLOW                                  │
└─────────────────────────────────────────────────────────────────────┘

  Request + Fingerprint
        │
        ▼
┌───────────────────┐
│   EXACT CACHE     │ ──── HIT ────► Response (0ms)
│   (SHA-256)       │                100% Confidence
└───────────────────┘
        │ MISS
        ▼
┌───────────────────┐          ┌───────────────────┐
│  SEMANTIC CACHE   │ ── HIT ─►│    VERIFIER       │
│  (Embeddings)     │          │  (Risk-Stratified)│
│  Similarity >0.92 │          └───────────────────┘
└───────────────────┘                   │
        │ MISS                    ┌─────┴─────┐
        │                         │           │
        ▼                      VALID       INVALID
   LLM Request                    │           │
        │                         ▼           ▼
        ▼                    Response    Regenerate
   Cache Store
```

### 8.2 Exact Cache Implementation

```python
import sqlite3
import hashlib
import json
from datetime import datetime, timedelta

class ExactCache:
    """
    Exakter Cache basierend auf SHA-256 Hash.
    Verwendet Working-Tree Fingerprint für Kontext-Sensitivität.
    """
    
    def __init__(self, db_path: str = "cache.sqlite"):
        self.db = sqlite3.connect(db_path, check_same_thread=False)
        self.db.execute("""
            CREATE TABLE IF NOT EXISTS exact_cache (
                cache_key TEXT PRIMARY KEY,
                query_hash TEXT,
                fingerprint TEXT,
                response TEXT,
                response_type TEXT,
                created_at TIMESTAMP,
                expires_at TIMESTAMP,
                hit_count INTEGER DEFAULT 0,
                last_hit_at TIMESTAMP
            )
        """)
        self.db.execute("""
            CREATE INDEX IF NOT EXISTS idx_expires ON exact_cache(expires_at)
        """)
        self.db.commit()
    
    def get_key(self, query: str, fingerprint: str) -> str:
        """Generiert Cache-Key aus Query + Fingerprint"""
        content = f"{query}|{fingerprint}"
        return hashlib.sha256(content.encode()).hexdigest()
    
    def get(self, query: str, fingerprint: str) -> Optional[dict]:
        """Sucht exakten Cache-Eintrag"""
        cache_key = self.get_key(query, fingerprint)
        now = datetime.now()
        
        row = self.db.execute("""
            SELECT response, response_type FROM exact_cache
            WHERE cache_key = ? AND expires_at > ?
        """, (cache_key, now)).fetchone()
        
        if row:
            # Hit-Counter aktualisieren
            self.db.execute("""
                UPDATE exact_cache 
                SET hit_count = hit_count + 1, last_hit_at = ?
                WHERE cache_key = ?
            """, (now, cache_key))
            self.db.commit()
            
            metrics.increment("cache_hit", tags={"type": "exact"})
            return {
                "response": json.loads(row[0]),
                "response_type": row[1],
                "cache_type": "exact"
            }
        
        metrics.increment("cache_miss", tags={"type": "exact"})
        return None
    
    def set(self, query: str, fingerprint: str, response: dict, 
            response_type: str, ttl_seconds: int):
        """Speichert Response im Cache"""
        cache_key = self.get_key(query, fingerprint)
        query_hash = hashlib.sha256(query.encode()).hexdigest()[:16]
        now = datetime.now()
        expires_at = now + timedelta(seconds=ttl_seconds)
        
        self.db.execute("""
            INSERT OR REPLACE INTO exact_cache 
            (cache_key, query_hash, fingerprint, response, response_type,
             created_at, expires_at, hit_count, last_hit_at)
            VALUES (?, ?, ?, ?, ?, ?, ?, 0, NULL)
        """, (
            cache_key, query_hash, fingerprint, json.dumps(response),
            response_type, now, expires_at
        ))
        self.db.commit()
    
    def invalidate_by_fingerprint(self, old_fingerprint: str):
        """Invalidiert alle Einträge mit altem Fingerprint"""
        self.db.execute("""
            DELETE FROM exact_cache WHERE fingerprint = ?
        """, (old_fingerprint,))
        self.db.commit()
    
    def cleanup_expired(self):
        """Entfernt abgelaufene Einträge"""
        self.db.execute("""
            DELETE FROM exact_cache WHERE expires_at < ?
        """, (datetime.now(),))
        self.db.commit()
```

### 8.3 Working-Tree Fingerprint

```python
import subprocess
import hashlib
from pathlib import Path

def build_working_tree_fingerprint(project_path: str) -> str:
    """
    Erstellt Fingerprint aus Git-Status + relevanten Dateien.
    Änderung im Fingerprint → Cache-Miss (korrekt!).
    """
    components = []
    
    try:
        # 1. Git HEAD Commit
        head = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=project_path,
            capture_output=True,
            text=True
        )
        if head.returncode == 0:
            components.append(f"head:{head.stdout.strip()[:12]}")
        
        # 2. Git Diff (staged + unstaged)
        diff = subprocess.run(
            ["git", "diff", "--stat", "HEAD"],
            cwd=project_path,
            capture_output=True,
            text=True
        )
        if diff.stdout.strip():
            diff_hash = hashlib.sha256(diff.stdout.encode()).hexdigest()[:8]
            components.append(f"diff:{diff_hash}")
        
        # 3. Aktive Dateien (die gerade bearbeitet werden)
        # Diese Info kommt vom OpenClaw Client
        active_files = get_active_files_from_context()
        if active_files:
            files_hash = hashlib.sha256(
                "|".join(sorted(active_files)).encode()
            ).hexdigest()[:8]
            components.append(f"active:{files_hash}")
        
        # 4. Package-Lock (Dependency-Änderungen)
        for lockfile in ["package-lock.json", "yarn.lock", "Cargo.lock", "poetry.lock"]:
            lockpath = Path(project_path) / lockfile
            if lockpath.exists():
                lock_hash = hashlib.sha256(
                    lockpath.read_bytes()
                ).hexdigest()[:8]
                components.append(f"lock:{lock_hash}")
                break
        
    except Exception as e:
        log.warning(f"Fingerprint partial failure: {e}")
        components.append(f"error:{str(e)[:20]}")
    
    return "|".join(components) if components else "empty"
```

### 8.4 Semantic Cache

```python
import numpy as np
from sqlite_utils import Database

class SemanticCache:
    """
    Semantischer Cache mit Embedding-Similarity.
    Verwendet lokale BM25 + Remote Embeddings (Fallback).
    """
    
    def __init__(self, db_path: str = "semantic_cache.sqlite", 
                 similarity_threshold: float = 0.92):
        self.db = Database(db_path)
        self.threshold = similarity_threshold
        
        # Tabelle für Embeddings
        self.db.execute("""
            CREATE TABLE IF NOT EXISTS semantic_cache (
                id INTEGER PRIMARY KEY,
                query TEXT,
                query_embedding BLOB,
                fingerprint TEXT,
                response TEXT,
                response_type TEXT,
                created_at TIMESTAMP,
                hit_count INTEGER DEFAULT 0
            )
        """)
    
    async def get(self, query: str, fingerprint: str) -> Optional[dict]:
        """Sucht semantisch ähnlichen Cache-Eintrag"""
        
        # Embedding für Query holen (mit Cache)
        query_embedding = await self.get_embedding(query)
        
        # Alle Einträge mit gleichem Fingerprint laden
        rows = list(self.db.execute("""
            SELECT id, query, query_embedding, response, response_type
            FROM semantic_cache 
            WHERE fingerprint = ?
        """, [fingerprint]).fetchall())
        
        if not rows:
            return None
        
        # Similarity berechnen
        best_match = None
        best_similarity = 0.0
        
        for row in rows:
            cached_embedding = np.frombuffer(row[2], dtype=np.float32)
            similarity = cosine_similarity(query_embedding, cached_embedding)
            
            if similarity > best_similarity and similarity >= self.threshold:
                best_similarity = similarity
                best_match = row
        
        if best_match:
            metrics.increment("cache_hit", tags={"type": "semantic"})
            return {
                "id": best_match[0],
                "original_query": best_match[1],
                "response": json.loads(best_match[3]),
                "response_type": best_match[4],
                "similarity": best_similarity,
                "cache_type": "semantic"
            }
        
        metrics.increment("cache_miss", tags={"type": "semantic"})
        return None
    
    async def get_embedding(self, text: str) -> np.ndarray:
        """Holt Embedding mit Query-Cache"""
        # Siehe Abschnitt 9.3 für Query Embedding Cache
        return await query_embedding_cache.get_or_compute(text)

def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    """Berechnet Cosine-Similarity zwischen zwei Vektoren"""
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))
```

### 8.5 Event-Driven Cache Invalidation

```python
# Git Hook: .git/hooks/post-commit
#!/bin/bash
# Invalidiert Code-bezogene Cache-Einträge nach Commit

COMMIT_HASH=$(git rev-parse HEAD)
CHANGED_FILES=$(git diff-tree --no-commit-id --name-only -r HEAD)

curl -X POST "http://localhost:8000/cache/invalidate" \
    -H "Authorization: Bearer $GATEWAY_SECRET" \
    -H "Content-Type: application/json" \
    -d "{
        \"event\": \"git_commit\",
        \"commit\": \"$COMMIT_HASH\",
        \"files\": $(echo "$CHANGED_FILES" | jq -R -s -c 'split("\n") | map(select(length > 0))')
    }"
```

```python
# Gateway Endpoint für Invalidation
@app.post("/cache/invalidate")
async def invalidate_cache(event: CacheInvalidationEvent, auth: AuthDep):
    """Invalidiert Cache basierend auf Events"""
    
    if event.event == "git_commit":
        # Alle code_suggestion Einträge invalidieren
        exact_cache.db.execute("""
            DELETE FROM exact_cache 
            WHERE response_type = 'code_suggestion'
        """)
        
        # Semantic Cache: Nur wenn betroffene Dateien
        if event.files:
            semantic_cache.invalidate_by_files(event.files)
        
        log.info(f"Cache invalidated for commit {event.commit[:8]}")
        return {"invalidated": True, "reason": "git_commit"}
    
    elif event.event == "manual":
        # Vollständige Invalidierung
        exact_cache.db.execute("DELETE FROM exact_cache")
        semantic_cache.db.execute("DELETE FROM semantic_cache")
        return {"invalidated": True, "reason": "manual_full"}
    
    return {"invalidated": False, "reason": "unknown_event"}
```

### 8.6 Adaptive TTLs

```python
def get_adaptive_ttl(response_type: str, hit_count: int = 0) -> int:
    """
    Berechnet TTL basierend auf Response-Type und Popularität.
    Populäre Einträge leben länger, ungenutzte werden schneller gelöscht.
    """
    base_ttls = {
        "explanation_generic": 7 * 24 * 3600,    # 7 Tage
        "explanation_contextual": 24 * 3600,      # 24 Stunden
        "code_suggestion": 0,                     # Bei Commit invalid
        "code_review": 12 * 3600,                 # 12 Stunden
        "command_execution": 3600,                # 1 Stunde
        "documentation": 24 * 3600,               # 24 Stunden
    }
    
    base = base_ttls.get(response_type, 3600)
    
    # Popularitäts-Multiplikator
    if hit_count >= 10:
        return int(base * 2)    # Sehr populär → 2x TTL
    elif hit_count >= 5:
        return int(base * 1.5)  # Populär → 1.5x TTL
    elif hit_count == 0:
        return int(base * 0.5)  # Ungenutzt → 0.5x TTL
    
    return base
```

---

## 9. Hybrid Retrieval (BM25 + Embeddings)

### 9.1 Warum Hybrid?

| Methode | Stärken | Schwächen |
|---------|---------|-----------|
| **BM25 (lokal)** | Exakte Begriffe, IDs, Error-Codes | Keine Semantik |
| **Embeddings (remote)** | Semantische Ähnlichkeit | Kosten, Latenz, Privacy |

**Lösung:** BM25 als Fast-Path für 80% der Queries, Remote Embeddings nur als Fallback.

### 9.2 BM25 mit SQLite FTS5

```python
class BM25Search:
    """Lokale Volltextsuche mit SQLite FTS5"""
    
    def __init__(self, db_path: str = "search.sqlite"):
        self.db = sqlite3.connect(db_path)
        self.db.execute("""
            CREATE VIRTUAL TABLE IF NOT EXISTS documents USING fts5(
                doc_id,
                title,
                content,
                doc_type,
                tokenize='porter unicode61'
            )
        """)
    
    def search(self, query: str, limit: int = 10) -> list[dict]:
        """BM25-Suche mit Relevanz-Score"""
        results = self.db.execute("""
            SELECT doc_id, title, content, doc_type,
                   bm25(documents) as score
            FROM documents
            WHERE documents MATCH ?
            ORDER BY score
            LIMIT ?
        """, (self._prepare_query(query), limit)).fetchall()
        
        return [
            {
                "doc_id": r[0],
                "title": r[1],
                "content": r[2][:500],  # Truncate
                "doc_type": r[3],
                "score": abs(r[4])  # BM25 returns negative scores
            }
            for r in results
        ]
    
    def _prepare_query(self, query: str) -> str:
        """Bereitet Query für FTS5 vor"""
        # Escape special characters
        query = re.sub(r'[^\w\s-]', ' ', query)
        # Tokenize und mit OR verbinden
        tokens = query.split()
        return " OR ".join(tokens)
```

### 9.3 Query Embedding Cache (NEU in v1.3)

```python
class QueryEmbeddingCache:
    """
    Query-Vektoren sind stabil → Cache für 30 Tage.
    Reduziert Remote-Embedding-Calls um 60-70%.
    """
    
    def __init__(self, db_path: str = "query_embeddings.sqlite"):
        self.db = sqlite3.connect(db_path)
        self.db.execute("""
            CREATE TABLE IF NOT EXISTS query_embeddings (
                query_hash TEXT PRIMARY KEY,
                query_normalized TEXT,
                embedding BLOB,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """)
        self.db.commit()
    
    async def get_or_compute(self, query: str) -> np.ndarray:
        """Holt Embedding aus Cache oder berechnet neu"""
        
        # Normalisiere Query
        normalized = self._normalize(query)
        query_hash = hashlib.sha256(normalized.encode()).hexdigest()[:16]
        
        # Check Cache
        row = self.db.execute("""
            SELECT embedding FROM query_embeddings WHERE query_hash = ?
        """, (query_hash,)).fetchone()
        
        if row:
            metrics.increment("query_embedding_cache_hit")
            return np.frombuffer(row[0], dtype=np.float32)
        
        # Compute + Store
        embedding = await self._compute_embedding(query)
        self.db.execute("""
            INSERT OR REPLACE INTO query_embeddings (query_hash, query_normalized, embedding)
            VALUES (?, ?, ?)
        """, (query_hash, normalized, embedding.tobytes()))
        self.db.commit()
        
        metrics.increment("query_embedding_cache_miss")
        return embedding
    
    def _normalize(self, query: str) -> str:
        """Normalisiert Query für Cache-Key"""
        # Lowercase, strip, collapse whitespace
        query = query.lower().strip()
        query = re.sub(r'\s+', ' ', query)
        # Entferne variable IDs (UUIDs, SHAs, etc.)
        query = re.sub(r'[a-f0-9]{8,}', '<ID>', query)
        return query
    
    async def _compute_embedding(self, text: str) -> np.ndarray:
        """Berechnet Embedding via OpenAI API"""
        async with httpx.AsyncClient() as client:
            response = await client.post(
                "https://api.openai.com/v1/embeddings",
                headers={"Authorization": f"Bearer {OPENAI_API_KEY}"},
                json={
                    "model": "text-embedding-3-small",
                    "input": text
                }
            )
            data = response.json()
            return np.array(data["data"][0]["embedding"], dtype=np.float32)
    
    def cleanup_old(self, days: int = 30):
        """Entfernt Einträge älter als X Tage"""
        self.db.execute("""
            DELETE FROM query_embeddings 
            WHERE created_at < datetime('now', ?)
        """, (f'-{days} days',))
        self.db.commit()

# Global instance
query_embedding_cache = QueryEmbeddingCache()
```

### 9.4 Fast-Path Detection (NEU in v1.3)

```python
# Patterns die BM25 gut matcht
TECHNICAL_PATTERNS = [
    r'[A-Z]{2,}_[A-Z_]+',           # ERROR_CODE, ECONNREFUSED
    r'[a-f0-9]{7,40}',               # Git SHA, UUIDs
    r'/[\w/-]+\.\w+',                # File paths
    r'\d+\.\d+\.\d+',                # Versions, IPs
    r'port\s*\d+',                   # Port numbers
    r'line\s*\d+',                   # Line numbers
    r'npm\s+ERR',                    # npm errors
    r'Error:\s+\w+',                 # Generic errors
    r'Exception\w*:',                # Exceptions
]

def is_technical_query(query: str) -> bool:
    """Erkennt technische Queries die BM25 gut findet"""
    return any(re.search(p, query, re.IGNORECASE) for p in TECHNICAL_PATTERNS)

async def hybrid_retrieval(query: str, project_context: dict) -> list[dict]:
    """
    Hybrid-Retrieval: BM25 Fast-Path + Remote Embeddings Fallback.
    """
    
    # FAST PATH: Technische Queries → nur BM25
    if is_technical_query(query):
        bm25_results = bm25_search.search(query, limit=5)
        
        if bm25_results and bm25_results[0]["score"] > 8.0:
            log.debug(f"Fast-path: BM25 hit for technical query")
            metrics.increment("retrieval_fast_path")
            return bm25_results
    
    # SLOW PATH: Semantische Queries → Hybrid
    bm25_results = bm25_search.search(query, limit=10)
    
    # Remote Embeddings nur wenn BM25 schlecht
    if not bm25_results or bm25_results[0]["score"] < 5.0:
        # Content-Filter vor Remote (Privacy)
        if not contains_sensitive_content(query):
            embedding_results = await embedding_search(query, limit=5)
            return merge_and_rerank(bm25_results, embedding_results)
    
    return bm25_results

def contains_sensitive_content(text: str) -> bool:
    """Prüft ob Text sensible Daten enthält → kein Remote Embedding"""
    sensitive_patterns = [
        r'api[_-]?key\s*[=:]',
        r'password\s*[=:]',
        r'secret\s*[=:]',
        r'-----BEGIN',
        r'/etc/',
        r'\.env',
        r'\.ssh/',
    ]
    return any(re.search(p, text, re.I) for p in sensitive_patterns)
```

---

## 10. Anthropic Prompt Caching

### 10.1 Wie es funktioniert

Anthropic cached automatisch identische Prompt-Prefixe. Der System-Prompt wird nur beim ersten Call verarbeitet, danach aus dem Cache geladen.

**Preise:**
| Vorgang | Preis (Sonnet) | vs. Normal |
|---------|----------------|------------|
| Schreiben ins Cache | $3.75/1M | +25% |
| Lesen aus Cache | $0.30/1M | **-90%** |
| Normal (kein Cache) | $3.00/1M | Baseline |

### 10.2 Implementation

```python
import anthropic

client = anthropic.Anthropic()

# Statischer System-Prompt (wird gecached)
STATIC_SYSTEM_PROMPT = """Du bist ein Senior Software Engineer mit Expertise in:
- Web Development (React, Node.js, Python)
- DevOps (Docker, Kubernetes, CI/CD)
- Datenbanken (PostgreSQL, MongoDB, Redis)

Dein Kommunikationsstil:
- Präzise und technisch korrekt
- Code-Beispiele immer als vollständige, lauffähige Snippets
- Bei Code-Änderungen: Unified Diff Format
- Bei Unsicherheit: Explizit kommunizieren

Ausgabeformat für Code-Änderungen:
```diff
--- a/path/to/file
+++ b/path/to/file
@@ -10,5 +10,7 @@
 context line
-old line
+new line
 context line
```

[... weitere 2500 Tokens Anweisungen ...]
"""

async def call_premium_with_caching(
    user_query: str,
    context: dict,
    max_tokens: int = 4096
) -> dict:
    """
    Premium-Call mit Prompt Caching.
    System-Prompt wird nach erstem Call gecached (90% Rabatt).
    """
    
    # Dynamischer Kontext in User-Message (NICHT in System-Prompt!)
    user_message = f"""
Projekt-Kontext:
- Pfad: {context.get('project_path', 'N/A')}
- Framework: {context.get('framework', 'N/A')}
- Git-Status: {context.get('git_status', 'clean')}

Aktive Dateien:
{context.get('active_files_summary', 'Keine')}

Anfrage:
{user_query}
"""
    
    response = client.messages.create(
        model="claude-sonnet-4-20250514",
        max_tokens=max_tokens,
        system=[
            {
                "type": "text",
                "text": STATIC_SYSTEM_PROMPT,
                "cache_control": {"type": "ephemeral"}  # WICHTIG!
            }
        ],
        messages=[
            {"role": "user", "content": user_message}
        ]
    )
    
    # Cache-Status loggen
    usage = response.usage
    if hasattr(usage, 'cache_read_input_tokens'):
        cache_read = usage.cache_read_input_tokens
        cache_write = usage.cache_creation_input_tokens
        log.info(f"Prompt cache: read={cache_read}, write={cache_write}")
        metrics.gauge("prompt_cache_read_tokens", cache_read)
    
    return {
        "content": response.content[0].text,
        "usage": {
            "input_tokens": usage.input_tokens,
            "output_tokens": usage.output_tokens
        }
    }
```

### 10.3 Kosten-Rechnung

```python
# Annahmen:
# - 30 Premium-Requests/Tag
# - 3000 Token System-Prompt
# - 1000 Token User-Message (variabel, nicht gecached)
# - 2000 Token Output

# OHNE Prompt Caching:
system_cost = 3000 * (3.00 / 1_000_000)  # = $0.009 pro Request
user_cost = 1000 * (3.00 / 1_000_000)    # = $0.003 pro Request
output_cost = 2000 * (15.00 / 1_000_000) # = $0.030 pro Request
total_per_request = 0.009 + 0.003 + 0.030  # = $0.042

monthly_requests = 30 * 30  # = 900
monthly_cost_without_cache = 900 * 0.042  # = $37.80/Monat

# MIT Prompt Caching:
# Erster Request: Cache-Write ($3.75/1M)
first_request_system = 3000 * (3.75 / 1_000_000)  # = $0.01125

# Folgende Requests: Cache-Read ($0.30/1M)
cached_system_cost = 3000 * (0.30 / 1_000_000)  # = $0.0009 pro Request

# Monatliche Kosten
first_day_cost = 0.01125 + (29 * 0.0009) + (30 * (0.003 + 0.030))
# ≈ $1.02 (System) + $0.99 (User+Output)

monthly_cost_with_cache = (
    0.01125 +                    # Erste Cache-Write
    (899 * 0.0009) +             # Cache-Reads
    (900 * 0.003) +              # User-Messages
    (900 * 0.030)                # Outputs
)
# = $0.01 + $0.81 + $2.70 + $27.00 = $30.52/Monat

# Ersparnis: $37.80 - $30.52 = $7.28/Monat (19%)
# Bei längerem System-Prompt (6000 Tokens): Ersparnis verdoppelt sich!
```

### 10.4 Best Practices

```python
# ✅ RICHTIG: Statischer System-Prompt
SYSTEM_PROMPT = "Du bist ein Assistent für Web-Development..."

# ❌ FALSCH: Dynamische Inhalte im System-Prompt (Cache-Miss!)
SYSTEM_PROMPT = f"Heute ist {date.today()}. Du bist ein Assistent..."

# ✅ RICHTIG: Dynamisches in User-Message
user_message = f"Kontext: Heute ist {date.today()}\n\nFrage: {query}"

# ✅ RICHTIG: Lange, stabile Anweisungen im System-Prompt
# ❌ FALSCH: Kurze System-Prompts (wenig Cache-Benefit)
```

---

## 11. Context Budgeting & Compression

### 11.1 Token-Budget pro Tier (NEU in v1.3)

| Tier | Max Input | Max Output | Bei Überschreitung |
|------|-----------|------------|-------------------|
| **Local** | 4.000 | 2.000 | Truncate + Warning |
| **Cheap** | 8.000 | 4.000 | Truncate + Warning |
| **Premium** | 16.000 | 8.000 | Compress + Summarize |

### 11.2 Context Budget Implementation

```python
import tiktoken

# Token-Encoder für Claude (approximiert mit cl100k_base)
encoder = tiktoken.get_encoding("cl100k_base")

def estimate_tokens(text: str) -> int:
    """Schätzt Token-Anzahl"""
    return len(encoder.encode(text))

class ContextBudget:
    """Begrenzt Context-Größe pro Tier"""
    
    LIMITS = {
        "local":   {"input": 4000,  "output": 2000},
        "cheap":   {"input": 8000,  "output": 4000},
        "premium": {"input": 16000, "output": 8000},
    }
    
    def apply(self, tier: str, context: dict) -> dict:
        """Wendet Budget auf Context an"""
        limits = self.LIMITS[tier]
        
        total_tokens = 0
        budgeted_context = {}
        
        # 1. Query immer komplett
        query = context.get("query", "")
        query_tokens = estimate_tokens(query)
        total_tokens += query_tokens
        budgeted_context["query"] = query
        
        # 2. System-Prompt (bei Premium gecached, zählt trotzdem)
        system_tokens = estimate_tokens(context.get("system_prompt", ""))
        total_tokens += system_tokens
        
        remaining = limits["input"] - total_tokens
        
        # 3. Dateien nach Relevanz
        files = context.get("files", [])
        budgeted_files = []
        
        for file in sorted(files, key=lambda f: f.get("relevance", 0), reverse=True):
            file_tokens = estimate_tokens(file.get("content", ""))
            
            if file_tokens <= remaining:
                budgeted_files.append(file)
                remaining -= file_tokens
            elif remaining > 500:
                # Excerpt statt volle Datei
                excerpt = self._extract_relevant(file, context["query"])
                excerpt_tokens = estimate_tokens(excerpt)
                if excerpt_tokens <= remaining:
                    budgeted_files.append({
                        **file,
                        "content": excerpt,
                        "truncated": True
                    })
                    remaining -= excerpt_tokens
        
        budgeted_context["files"] = budgeted_files
        
        # 4. Logs komprimieren
        if "logs" in context:
            compressed_logs = self._compress_logs(
                context["logs"], 
                max_tokens=min(remaining, 2000)
            )
            budgeted_context["logs"] = compressed_logs
        
        return budgeted_context
    
    def _extract_relevant(self, file: dict, query: str) -> str:
        """Extrahiert relevante Code-Abschnitte"""
        content = file.get("content", "")
        
        # Finde relevante Funktionen/Klassen
        # (vereinfachte Implementierung)
        lines = content.split("\n")
        relevant_lines = []
        
        for i, line in enumerate(lines):
            # Suche nach Query-Keywords
            if any(kw.lower() in line.lower() for kw in query.split()):
                # Kontext: 5 Zeilen davor/danach
                start = max(0, i - 5)
                end = min(len(lines), i + 6)
                relevant_lines.extend(lines[start:end])
                relevant_lines.append("...")
        
        if relevant_lines:
            return "\n".join(relevant_lines[:100])  # Max 100 Zeilen
        
        # Fallback: Anfang der Datei
        return "\n".join(lines[:50])
    
    def _compress_logs(self, logs: str, max_tokens: int) -> str:
        """Komprimiert Logs für Context"""
        return compress_logs_for_context(logs, max_tokens)
```

### 11.3 Log/Trace Compression Pipeline

```python
import re

def compress_logs_for_context(logs: str, max_tokens: int = 2000) -> str:
    """
    Komprimiert Logs durch:
    1. Duplikat-Entfernung
    2. Timestamp-Normalisierung
    3. Stack-Trace-Kürzung
    4. Token-Budget-Trimming
    """
    lines = logs.split("\n")
    
    # 1. Duplikate entfernen (häufig bei Retry-Loops)
    seen = set()
    unique_lines = []
    
    for line in lines:
        # Normalisiere: Entferne Timestamps, IDs
        normalized = re.sub(
            r'\d{4}-\d{2}-\d{2}[T ]\d{2}:\d{2}:\d{2}[.\d]*Z?',
            '<TIME>',
            line
        )
        normalized = re.sub(r'[a-f0-9]{8,}', '<ID>', normalized)
        
        if normalized not in seen:
            seen.add(normalized)
            unique_lines.append(line)
    
    # 2. Stack-Traces kürzen (nur erste + letzte 3 Frames)
    compressed = compress_stacktraces(unique_lines)
    
    # 3. Auf Token-Budget trimmen
    result = "\n".join(compressed)
    while estimate_tokens(result) > max_tokens and compressed:
        compressed = compressed[1:]  # Älteste Zeile entfernen
        result = "\n".join(compressed)
    
    return result

def compress_stacktraces(lines: list[str]) -> list[str]:
    """Kürzt Stack-Traces auf wesentliche Frames"""
    result = []
    in_stacktrace = False
    stack_buffer = []
    
    for line in lines:
        # Erkennung: Stack-Trace Anfang
        if re.match(r'\s*(at |File "|Traceback)', line):
            in_stacktrace = True
            stack_buffer.append(line)
            continue
        
        # Ende des Stack-Traces
        if in_stacktrace and not re.match(r'\s*(at |File "|\s+\^)', line):
            in_stacktrace = False
            
            # Nur erste 3 + letzte 3 Frames
            if len(stack_buffer) > 6:
                result.extend(stack_buffer[:3])
                result.append(f"    ... ({len(stack_buffer) - 6} frames omitted)")
                result.extend(stack_buffer[-3:])
            else:
                result.extend(stack_buffer)
            
            stack_buffer = []
        
        if in_stacktrace:
            stack_buffer.append(line)
        else:
            result.append(line)
    
    return result
```

---

## 12. Capability-based Tools

### 12.1 Drei-Zonen-Modell

| Zone | Pfade | KI-Autonomie | Confirm? |
|------|-------|--------------|----------|
| **App/Projekt** | /srv/projects/*, dist, .cache | ✅ Voll | Nein |
| **User/Runtime** | $HOME/.local, venv, docker | ⚠️ Eingeschränkt | Bei Schreiben |
| **System** | /etc, /usr, /var/lib, systemd | ❌ Blockiert | Immer |

### 12.2 Tool-Definitionen

```python
from pydantic import BaseModel, Field
from typing import Literal, Optional
from pathlib import Path

# Definierte Roots für verschiedene Operationen
DELETABLE_ROOTS = [
    "/srv/projects/*/tmp/",
    "/srv/projects/*/dist/",
    "/srv/projects/*/.cache/",
    "/srv/projects/*/node_modules/.cache/",
    "/srv/projects/*/coverage/",
]

FORBIDDEN_PATHS = [
    "**/.env",
    "**/*.pem",
    "**/*.key",
    "/etc/**",
    "/.ssh/**",
    "**/id_rsa*",
]

class FSDeleteInput(BaseModel):
    """Input für fs_delete Tool"""
    path: str = Field(..., description="Pfad zum Löschen")
    recursive: bool = Field(False, description="Rekursiv löschen")
    dry_run: bool = Field(True, description="Nur simulieren")

class FSDeleteResult(BaseModel):
    """Ergebnis von fs_delete"""
    success: bool
    deleted_count: int
    total_size_bytes: int
    sample_files: list[str]
    dry_run: bool
    error: Optional[str] = None

async def fs_delete(input: FSDeleteInput) -> FSDeleteResult:
    """
    Sicheres Löschen mit Trash-Retention.
    Nur in erlaubten Pfaden, mit Audit-Log.
    """
    path = Path(input.path).resolve()
    
    # 1. Prüfe gegen Forbidden Paths
    for pattern in FORBIDDEN_PATHS:
        if path.match(pattern):
            return FSDeleteResult(
                success=False,
                deleted_count=0,
                total_size_bytes=0,
                sample_files=[],
                dry_run=input.dry_run,
                error=f"Forbidden path: {pattern}"
            )
    
    # 2. Prüfe gegen Deletable Roots
    allowed = False
    for root_pattern in DELETABLE_ROOTS:
        if path.match(root_pattern.replace("*", "**")):
            allowed = True
            break
    
    if not allowed:
        return FSDeleteResult(
            success=False,
            deleted_count=0,
            total_size_bytes=0,
            sample_files=[],
            dry_run=input.dry_run,
            error=f"Path not in deletable roots"
        )
    
    # 3. Sammle Dateien
    if input.recursive and path.is_dir():
        files = list(path.rglob("*"))
    elif path.is_file():
        files = [path]
    else:
        files = list(path.glob("*"))
    
    total_size = sum(f.stat().st_size for f in files if f.is_file())
    sample = [str(f) for f in files[:5]]
    
    # 4. Dry-Run oder echtes Löschen
    if input.dry_run:
        return FSDeleteResult(
            success=True,
            deleted_count=len(files),
            total_size_bytes=total_size,
            sample_files=sample,
            dry_run=True
        )
    
    # 5. Move to Trash (7 Tage Retention)
    trash_path = Path(f"/srv/trash/{datetime.now().strftime('%Y%m%d')}")
    trash_path.mkdir(parents=True, exist_ok=True)
    
    import shutil
    shutil.move(str(path), str(trash_path / path.name))
    
    # 6. Audit Log
    log.info(f"fs_delete: {path} → {trash_path}, {len(files)} files, {total_size} bytes")
    
    return FSDeleteResult(
        success=True,
        deleted_count=len(files),
        total_size_bytes=total_size,
        sample_files=sample,
        dry_run=False
    )
```

### 12.3 Cleanup-Policies

```python
CLEANUP_POLICIES = {
    "node_cache_v1": {
        "description": "Node.js Build-Cache bereinigen",
        "targets": [
            "node_modules/.cache/**",
            ".next/cache/**",
            "dist/**/*.map",
        ],
        "max_size_mb": 5000,
        "min_age_days": 0,
    },
    "build_artifacts_v1": {
        "description": "Build-Artefakte älter als 7 Tage",
        "targets": [
            "dist/**",
            "build/**",
            "coverage/**",
            "**/*.log",
        ],
        "max_size_mb": 10000,
        "min_age_days": 7,
    },
    "docker_cache_v1": {
        "description": "Docker Build-Cache und dangling Images",
        "commands": [
            "docker image prune -f",
            "docker builder prune -f --filter until=168h",
        ],
        "max_size_mb": 20000,
        "requires_confirm": True,  # Immer User-Bestätigung!
    },
}

async def cleanup_project(project_id: str, policy_name: str, dry_run: bool = True):
    """Führt Cleanup-Policy aus"""
    
    if policy_name not in CLEANUP_POLICIES:
        raise ValueError(f"Unknown policy: {policy_name}")
    
    policy = CLEANUP_POLICIES[policy_name]
    project_path = Path(f"/srv/projects/{project_id}")
    
    if not project_path.exists():
        raise ValueError(f"Project not found: {project_id}")
    
    results = []
    
    for target in policy.get("targets", []):
        matched_files = list(project_path.glob(target))
        
        for f in matched_files:
            if policy.get("min_age_days", 0) > 0:
                age_days = (datetime.now() - datetime.fromtimestamp(f.stat().st_mtime)).days
                if age_days < policy["min_age_days"]:
                    continue
            
            results.append({
                "path": str(f),
                "size": f.stat().st_size if f.is_file() else 0,
                "action": "delete" if not dry_run else "would_delete"
            })
            
            if not dry_run:
                await fs_delete(FSDeleteInput(path=str(f), recursive=True, dry_run=False))
    
    return {
        "policy": policy_name,
        "project": project_id,
        "dry_run": dry_run,
        "files_affected": len(results),
        "total_size_mb": sum(r["size"] for r in results) / (1024 * 1024),
        "details": results[:20]  # Nur erste 20 für Übersicht
    }
```

---

## 13. Deterministisches Patching

### 13.1 Warum Deterministisch?

LLMs sind nicht-deterministisch → gleiche Anfrage kann unterschiedliche Patches erzeugen.
**Problem:** Bei erneutem Ausführen passt der Patch evtl. nicht mehr zum Code.

**Lösung:**
1. **Fingerprint** erfasst den exakten Code-Zustand
2. **Verifier** prüft ob Patch noch passt
3. **Transaction-Pattern** macht Patches atomar

### 13.2 Patch-Format: Unified Diff

```python
from dataclasses import dataclass

@dataclass
class PatchFile:
    path: str
    old_content: str
    new_content: str
    hunks: list[dict]

@dataclass
class Patch:
    files: list[PatchFile]
    description: str
    author: str = "llm-gateway"

def parse_unified_diff(diff_text: str) -> Patch:
    """Parst Unified Diff in strukturiertes Format"""
    import unidiff
    
    patchset = unidiff.PatchSet.from_string(diff_text)
    files = []
    
    for patched_file in patchset:
        hunks = []
        for hunk in patched_file:
            hunks.append({
                "source_start": hunk.source_start,
                "source_length": hunk.source_length,
                "target_start": hunk.target_start,
                "target_length": hunk.target_length,
                "content": str(hunk)
            })
        
        files.append(PatchFile(
            path=patched_file.path,
            old_content="",  # Wird später geladen
            new_content="",
            hunks=hunks
        ))
    
    return Patch(files=files, description="")
```

### 13.3 Transaction-Pattern

```python
import shutil
from contextlib import contextmanager

@contextmanager
def patch_transaction(project_path: str):
    """
    Atomares Anwenden von Patches mit Rollback.
    Bei Fehler: Alle Änderungen werden rückgängig gemacht.
    """
    backup_path = f"{project_path}/.patch-backup-{datetime.now().timestamp()}"
    changed_files = []
    
    try:
        yield PatchContext(project_path, backup_path, changed_files)
        
        # Commit: Backup löschen
        if os.path.exists(backup_path):
            shutil.rmtree(backup_path)
        
        log.info(f"Patch committed: {len(changed_files)} files changed")
        
    except Exception as e:
        # Rollback: Alle Dateien aus Backup wiederherstellen
        log.error(f"Patch failed, rolling back: {e}")
        
        for file_path in changed_files:
            backup_file = os.path.join(backup_path, os.path.relpath(file_path, project_path))
            if os.path.exists(backup_file):
                shutil.copy2(backup_file, file_path)
        
        if os.path.exists(backup_path):
            shutil.rmtree(backup_path)
        
        raise

class PatchContext:
    def __init__(self, project_path, backup_path, changed_files):
        self.project_path = project_path
        self.backup_path = backup_path
        self.changed_files = changed_files
    
    def modify_file(self, relative_path: str, new_content: str):
        """Modifiziert Datei mit automatischem Backup"""
        full_path = os.path.join(self.project_path, relative_path)
        backup_file = os.path.join(self.backup_path, relative_path)
        
        # Backup erstellen
        os.makedirs(os.path.dirname(backup_file), exist_ok=True)
        if os.path.exists(full_path):
            shutil.copy2(full_path, backup_file)
        
        # Datei ändern
        with open(full_path, 'w') as f:
            f.write(new_content)
        
        self.changed_files.append(full_path)

# Verwendung
async def apply_patch_safely(project_path: str, patch: Patch):
    """Wendet Patch mit Transaction-Pattern an"""
    
    with patch_transaction(project_path) as ctx:
        for file_change in patch.files:
            # Lade aktuellen Inhalt
            full_path = os.path.join(project_path, file_change.path)
            
            if os.path.exists(full_path):
                with open(full_path) as f:
                    current_content = f.read()
            else:
                current_content = ""
            
            # Wende Hunks an
            new_content = apply_hunks(current_content, file_change.hunks)
            
            # Schreibe mit Backup
            ctx.modify_file(file_change.path, new_content)
    
    return {"success": True, "files_changed": len(patch.files)}
```

---

## 14. Patch Risk Score

### 14.1 Warum Risk Score statt fixer Limits?

v1.2 nutzte fixes Maximum (300 Lines). Problem: 
- Blockt legitime Refactorings
- Erlaubt gefährliche kleine Änderungen (z.B. in auth/)

**v1.3 Lösung:** Gewichteter Risk Score basierend auf:
- Pfad-Risiko (auth/, payment/, etc.)
- Dateityp-Risiko (.sql, .env, etc.)
- Change-Ratio (% der Datei geändert)

### 14.2 Risk Score Implementation

```python
HIGH_RISK_PATHS = {
    "auth/": 10,
    "authentication/": 10,
    "payment/": 10,
    "payments/": 10,
    "billing/": 8,
    "migrations/": 8,
    "infra/": 8,
    "infrastructure/": 8,
    "config/": 6,
    "security/": 10,
    "admin/": 6,
    ".github/workflows/": 6,
}

HIGH_RISK_EXTENSIONS = {
    ".sql": 8,
    ".env": 10,
    ".env.local": 10,
    ".env.production": 10,
    ".pem": 10,
    ".key": 10,
    ".crt": 6,
    "Dockerfile": 6,
    "docker-compose.yml": 6,
    "docker-compose.yaml": 6,
    ".yml": 4,
    ".yaml": 4,
}

@dataclass
class PatchRiskResult:
    score: int
    reasons: list[str]
    allowed: bool
    requires_review: bool

def calculate_patch_risk(diff: Patch) -> PatchRiskResult:
    """
    Berechnet Risk Score für einen Patch.
    
    Score-Interpretation:
    - 0-7:   Low Risk → Auto-Apply erlaubt
    - 8-14:  Medium Risk → Apply mit Review-Flag
    - 15+:   High Risk → Blockiert
    """
    score = 0
    reasons = []
    
    for file in diff.files:
        path_lower = file.path.lower()
        
        # Pfad-Risiko
        for pattern, weight in HIGH_RISK_PATHS.items():
            if pattern in path_lower:
                score += weight
                reasons.append(f"high_risk_path:{pattern}")
        
        # Extension-Risiko
        for ext, weight in HIGH_RISK_EXTENSIONS.items():
            if path_lower.endswith(ext) or ext in path_lower:
                score += weight
                reasons.append(f"high_risk_extension:{ext}")
        
        # Change-Ratio Risiko
        if file.old_content:
            total_lines = len(file.old_content.split("\n"))
            changed_lines = sum(len(h.get("content", "").split("\n")) for h in file.hunks)
            
            if total_lines > 0:
                ratio = changed_lines / total_lines
                if ratio > 0.8 and changed_lines > 50:
                    score += 5
                    reasons.append(f"high_change_ratio:{ratio:.0%}")
                elif ratio > 0.5 and changed_lines > 100:
                    score += 3
                    reasons.append(f"medium_change_ratio:{ratio:.0%}")
    
    # Anzahl Dateien
    if len(diff.files) > 10:
        score += 3
        reasons.append(f"many_files:{len(diff.files)}")
    
    return PatchRiskResult(
        score=score,
        reasons=reasons,
        allowed=score < 15,
        requires_review=score >= 8
    )

def validate_patch_scope(diff: Patch) -> tuple[bool, str]:
    """
    Validiert Patch gegen Hard-Limits.
    Returns: (allowed, reason)
    """
    # Absolut verbotene Pfade
    forbidden = ["/etc/", "/.env", "/.ssh/", "*.pem", "*.key", "id_rsa"]
    
    for file in diff.files:
        for pattern in forbidden:
            if pattern.startswith("*"):
                if file.path.endswith(pattern[1:]):
                    return False, f"Forbidden extension: {pattern}"
            elif pattern in file.path:
                return False, f"Forbidden path: {pattern}"
    
    # Risk Score prüfen
    risk = calculate_patch_risk(diff)
    if not risk.allowed:
        return False, f"Risk score too high: {risk.score} ({', '.join(risk.reasons)})"
    
    return True, "ok"
```

---

## 15. Risk-Stratified Verifier

### 15.1 Warum nicht immer verifizieren?

v1.2 rief Haiku-Verifier bei **jedem** Semantic Cache Hit.
**Problem:** 
- Kosten (auch wenn gering: $0.0003/Call)
- Latenz (+300-500ms)
- Unnötig für Low-Risk Response-Types

**v1.3 Lösung:** Verifier nur bei tatsächlichem Risiko.

### 15.2 Verifier-Entscheidungsbaum

```python
def should_verify_cache_hit(
    response_type: str,
    similarity: float,
    fingerprint_changed: bool,
    file_touched: bool = False
) -> tuple[bool, str]:
    """
    Entscheidet ob Verifier nötig ist.
    
    Returns: (should_verify, reason)
    """
    
    # HIGH RISK: Immer verifizieren
    HIGH_RISK_TYPES = ["code_patch", "command_execution", "code_suggestion"]
    if response_type in HIGH_RISK_TYPES:
        return True, "high_risk_response_type"
    
    # Fingerprint unverändert → Sehr sicher
    if not fingerprint_changed:
        return False, "fingerprint_unchanged"
    
    # MEDIUM RISK: Nur bei niedriger Similarity oder File-Änderung
    MEDIUM_RISK_TYPES = ["explanation_contextual", "code_review"]
    if response_type in MEDIUM_RISK_TYPES:
        if similarity < 0.97:
            return True, "contextual_low_similarity"
        if file_touched:
            return True, "contextual_file_changed"
        return False, "contextual_high_similarity"
    
    # LOW RISK: Nie verifizieren
    LOW_RISK_TYPES = ["explanation_generic", "documentation"]
    if response_type in LOW_RISK_TYPES:
        return False, "generic_always_safe"
    
    # Unknown → Sicherheitshalber verifizieren
    return True, "unknown_response_type"
```

### 15.3 Verifier Implementation (JSON Output)

```python
async def verify_cache_hit(
    cached_response: dict,
    current_fingerprint: str,
    original_fingerprint: str,
    query: str
) -> dict:
    """
    Verifiziert ob gecachte Antwort noch gültig ist.
    
    Returns: {
        "verdict": "VALID" | "INVALID",
        "reason": str,
        "confidence": float,
        "suggestion": Optional[str]
    }
    """
    
    # Quick-Check: Fingerprint identisch → Immer valid
    if current_fingerprint == original_fingerprint:
        return {
            "verdict": "VALID",
            "reason": "fingerprint_unchanged",
            "confidence": 1.0
        }
    
    # Haiku-Verifier für komplexe Prüfung
    verification_prompt = f"""Prüfe ob diese Antwort noch korrekt ist:

URSPRÜNGLICHE ANFRAGE:
{query}

GECACHTE ANTWORT:
{json.dumps(cached_response.get('content', ''), indent=2)[:2000]}

KONTEXT-ÄNDERUNGEN:
- Alter Fingerprint: {original_fingerprint}
- Neuer Fingerprint: {current_fingerprint}

Antworte NUR mit JSON:
{{
    "verdict": "VALID" oder "INVALID",
    "reason": "context_unchanged|version_mismatch|missing_files|code_changed|security_risk",
    "confidence": 0.0-1.0,
    "suggestion": "Optional: Was neu generiert werden sollte"
}}"""

    response = await haiku_client.messages.create(
        model="claude-3-5-haiku-20241022",
        max_tokens=200,
        messages=[{"role": "user", "content": verification_prompt}]
    )
    
    try:
        result = json.loads(response.content[0].text)
        return {
            "verdict": result.get("verdict", "INVALID"),
            "reason": result.get("reason", "parse_error"),
            "confidence": float(result.get("confidence", 0.5)),
            "suggestion": result.get("suggestion")
        }
    except json.JSONDecodeError:
        # Fallback bei Parse-Fehler
        return {
            "verdict": "INVALID",
            "reason": "verifier_parse_error",
            "confidence": 0.0,
            "suggestion": "Regenerate response"
        }
```

### 15.4 Two-Stage Verifier (Optional)

```python
async def two_stage_verify(cached: dict, fingerprint_current: str, fingerprint_cached: str) -> dict:
    """
    Zweistufige Verifikation:
    1. Lokale Heuristiken (schnell, kostenlos)
    2. LLM-Verifier (nur wenn nötig)
    """
    
    # STAGE 1: Lokale Heuristiken
    
    # 1a. Fingerprint identisch → VALID
    if fingerprint_current == fingerprint_cached:
        return {"verdict": "VALID", "reason": "fingerprint_match", "stage": 1}
    
    # 1b. Nur Timestamp-Diff → wahrscheinlich VALID
    fp_current = parse_fingerprint(fingerprint_current)
    fp_cached = parse_fingerprint(fingerprint_cached)
    
    if fp_current.get("head") == fp_cached.get("head"):
        if fp_current.get("diff") == fp_cached.get("diff"):
            return {"verdict": "VALID", "reason": "same_head_and_diff", "stage": 1}
    
    # 1c. Response-Type prüfen
    if cached.get("response_type") == "explanation_generic":
        return {"verdict": "VALID", "reason": "generic_always_valid", "stage": 1}
    
    # STAGE 2: LLM-Verifier (nur für unsichere Fälle)
    return await verify_cache_hit(cached, fingerprint_current, fingerprint_cached, cached.get("query", ""))
```

---

## 16. Monitoring & KPIs

### 16.1 Vollständige KPI-Tabelle

| Metrik | Beschreibung | Ziel | Warnung | Kritisch | Aktion |
|--------|--------------|------|---------|----------|--------|
| `daily_cost_usd` | Tägliche API-Kosten | <$2 | >$5 | >$15 | Throttle/Kill |
| `groq_latency_p95_ms` | Router-Latenz (95. Perzentil) | <300 | >500 | >1000 | Fallback Haiku |
| `cache_hit_rate_total` | Gesamt-Cache-Hit-Rate | >40% | <25% | <15% | Cache-Config |
| `exact_cache_hit_rate` | Exact Cache Hit-Rate | >25% | <10% | <5% | Key-Format |
| `semantic_cache_hit_rate` | Semantic Cache Hit-Rate | >15% | <5% | <2% | Threshold |
| `premium_ratio` | Anteil Premium-Requests | <25% | >35% | >50% | Router tunen |
| `verifier_skip_rate` | Verifier-Skips (Risk-Based) | >50% | <30% | N/A | Risk-Config |
| `verifier_reject_rate` | Verifier-Rejections | <20% | >40% | >60% | Threshold |
| `context_compression_ratio` | Kontext-Kompression | >30% | <15% | N/A | Pipeline |
| `idempotency_hit_rate` | Doppelte Requests | <5% | >10% | >20% | Client-Bug? |
| `patch_success_rate` | Erfolgreiche Patches | >90% | <75% | <50% | Diff-Format |
| `patch_high_risk_rate` | High-Risk Patches | <10% | >20% | >30% | Review |
| `policy_block_rate` | Hard-Policy Blocks | Log | >5% | >10% | Injection? |
| `rate_limit_hits` | Rate-Limit Treffer | <5% | >15% | >25% | Limits |
| `cache_size_mb` | Cache-Größe | <500 | >800 | >1000 | Eviction |
| `query_embedding_cache_hit` | Query-Embedding Cache | >60% | <40% | <20% | TTL |

### 16.2 Alerting-Regeln

```python
ALERT_RULES = {
    "budget_soft": {
        "metric": "daily_cost_usd",
        "condition": "> 5.0",
        "severity": "warning",
        "action": "log_and_notify"
    },
    "budget_medium": {
        "metric": "daily_cost_usd",
        "condition": "> 15.0",
        "severity": "error",
        "action": "throttle_premium"
    },
    "budget_hard": {
        "metric": "daily_cost_usd",
        "condition": "> 50.0",
        "severity": "critical",
        "action": "kill_switch"
    },
    "groq_degraded": {
        "metric": "groq_latency_p95_ms",
        "condition": "> 1000",
        "severity": "error",
        "action": "fallback_to_haiku"
    },
    "cache_ineffective": {
        "metric": "cache_hit_rate_total",
        "condition": "< 0.15",
        "severity": "warning",
        "action": "review_cache_config"
    },
    "possible_attack": {
        "metric": "policy_block_rate",
        "condition": "> 0.10",
        "severity": "critical",
        "action": "review_and_block_client"
    },
    "storage_warning": {
        "metric": "cache_size_mb",
        "condition": "> 800",
        "severity": "warning",
        "action": "run_eviction"
    }
}
```

### 16.3 Prometheus Metrics

```python
from prometheus_client import Counter, Histogram, Gauge

# Counters
requests_total = Counter(
    "gateway_requests_total",
    "Total requests",
    ["tier", "cache_type", "response_type"]
)

policy_blocks = Counter(
    "gateway_policy_blocks_total",
    "Policy blocks",
    ["category", "pattern"]
)

# Histograms
request_latency = Histogram(
    "gateway_request_latency_seconds",
    "Request latency",
    ["tier"],
    buckets=[0.1, 0.25, 0.5, 1.0, 2.5, 5.0, 10.0]
)

# Gauges
daily_spend = Gauge(
    "gateway_daily_spend_usd",
    "Daily API spend in USD"
)

cache_size = Gauge(
    "gateway_cache_size_bytes",
    "Cache size in bytes",
    ["cache_type"]
)

active_requests = Gauge(
    "gateway_active_requests",
    "Currently processing requests"
)
```

---

## 17. Implementierungsplan

### 17.1 Phase 1: Infrastruktur + Basis (Tage 1-3)

| Tag | Aufgabe | Deliverable | Go/No-Go |
|-----|---------|-------------|----------|
| 1 | Hetzner CX22 bestellen, Ubuntu Setup, Firewall, SSH | Server erreichbar | SSH funktioniert |
| 2 | FastAPI Skeleton, Groq Router, Hard Policy Gate | `/health` + `/route` Endpoints | Routing funktioniert |
| 3 | Rate Limiting, Kill Switch, Idempotency | Budget-System aktiv | Cap bei $5 greift |

**Tag 1 Commands:**
```bash
# Hetzner Cloud Console: CX22 mit Ubuntu 24.04
# DNS: gateway.yourdomain.com → Server-IP

ssh root@<ip>
apt update && apt upgrade -y
apt install -y python3-pip python3-venv nginx certbot python3-certbot-nginx
ufw allow 22,80,443/tcp && ufw enable
```

**Tag 2 Skeleton:**
```python
# main.py
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

app = FastAPI(title="LLM Gateway", version="1.3.0")

@app.get("/health")
async def health():
    return {"status": "ok", "version": "1.3.0"}

@app.post("/v1/chat/completions")
async def chat_completions(request: ChatRequest):
    # 1. Hard Policy Gate
    violation = check_hard_policy(request.messages[-1].content)
    if violation:
        raise HTTPException(403, f"Policy violation: {violation.category}")
    
    # 2. Rate Limiting
    rate_result = rate_limiter.check("premium", estimate_tokens(request))
    if not rate_result[0]:
        raise HTTPException(429, f"Rate limit: {rate_result[1]}")
    
    # 3. Budget Guard
    budget_result = budget_guard.check("premium", 0.05)
    if not budget_result["allowed"]:
        raise HTTPException(429, f"Budget limit: {budget_result['reason']}")
    
    # 4. Route + Process
    ...
```

### 17.2 Phase 2: Caching + Premium (Tage 4-7)

| Tag | Aufgabe | Deliverable | Go/No-Go |
|-----|---------|-------------|----------|
| 4 | BM25 (FTS5), Query Embedding Cache | Retrieval funktioniert | Fast-Path aktiv |
| 5 | Anthropic Prompt Caching aktivieren | Cache-Headers in Logs | Cache-Read >0 |
| 6 | Semantic Cache + Risk-Stratified Verifier | Verifier-Skips in Logs | Skip-Rate >50% |
| 7 | Context Budgeting + Log Compression | Compression in Logs | Ratio >30% |

**Tag 5: Prompt Caching aktivieren**
```python
# Nur diese Änderung nötig:
response = client.messages.create(
    model="claude-sonnet-4-20250514",
    system=[{
        "type": "text",
        "text": STATIC_SYSTEM_PROMPT,
        "cache_control": {"type": "ephemeral"}  # <-- Diese Zeile!
    }],
    messages=[...]
)

# Logging zum Verifizieren:
usage = response.usage
log.info(f"Cache: read={usage.cache_read_input_tokens}, write={usage.cache_creation_input_tokens}")
```

### 17.3 Phase 3: Tools + Hardening (Tage 8-10)

| Tag | Aufgabe | Deliverable | Go/No-Go |
|-----|---------|-------------|----------|
| 8 | Capability Tools, Patch Risk Score | Tools funktionieren | High-Risk blocked |
| 9 | Monitoring Dashboard, Alerting | Grafana/Prometheus | Alerts funktionieren |
| 10 | 24h Lasttest, Kosten-Audit | Test-Report | <€1 für Test-Tag |

**Tag 10: Lasttest**
```bash
# Simuliere 500 Requests über 24h
for i in {1..500}; do
  curl -X POST https://gateway.yourdomain.com/v1/chat/completions \
    -H "Authorization: Bearer $TOKEN" \
    -d '{"messages":[{"role":"user","content":"Explain async/await in JavaScript"}]}'
  sleep 172  # ~500 Requests in 24h
done

# Erwartete Kosten:
# - 400 Cache-Hits → $0
# - 80 Cheap (Haiku) → $0.02
# - 20 Premium (mit Caching) → $0.60
# TOTAL: ~$0.62 für 500 Requests
```

### 17.4 Phase 4: Beta + Production (Tage 11-14)

| Tag | Aufgabe | Deliverable |
|-----|---------|-------------|
| 11 | Beta-Rollout (10% Traffic) | Monitoring aktiv |
| 12 | Feedback sammeln, Feintuning | Anpassungen dokumentiert |
| 13 | Rollout auf 50%, dann 100% | Production Traffic |
| 14 | Dokumentation, Runbook | Operations-Guide |

---

## 18. Kostenprognose

### 18.1 Detaillierte Aufschlüsselung

| Komponente | Berechnung | Monatlich |
|------------|------------|-----------|
| **Hetzner CX22** | €4,35 fix | €4,35 |
| **Groq Router** | 15.000 Req × 300 Tok × $0.05/1M | €0,25 |
| **Embeddings** | 5.000 Queries × $0.02/1K Tok | €1,00 |
| **Haiku (Cheap)** | 10.000 Req × 2K Tok × $0.25/1M | €5,00 |
| **Haiku (Verifier)** | 2.000 Calls × 500 Tok × $0.25/1M | €0,25 |
| **Sonnet (Premium)** | 3.000 Req × 4K Tok × $3/1M (Input) | €3,60 |
| **Sonnet (Premium)** | 3.000 Req × 2K Tok × $15/1M (Output) | €9,00 |
| **Prompt Cache Rabatt** | 3.000 Req × 3K Tok × ($3-$0.30)/1M | -€2,43 |
| **TOTAL** | | **€21,02** |

### 18.2 Szenario-Vergleich

| Szenario | Requests/Tag | Premium % | Cache-Hit % | Kosten/Monat |
|----------|--------------|-----------|-------------|--------------|
| **Light** | 100 | 15% | 50% | €12-15 |
| **Standard** | 500 | 20% | 40% | €20-25 |
| **Heavy** | 1000 | 25% | 35% | €40-50 |
| **Worst Case** | 2000 | 40% | 20% | €80-100 |

### 18.3 Break-Even-Analyse

```
Ohne Gateway (direkt Sonnet):
- 500 Req/Tag × 6K Tok × $18/1M = $162/Monat

Mit Gateway:
- ~$25/Monat

Ersparnis: $137/Monat = 85%
Break-Even: Ab 50 Requests/Tag
```

---

## 19. Changelog

### v1.3 (Februar 2026) - Cost-Optimized

**Infrastruktur:**
- ✅ Hetzner statt AWS (-€60/Monat)
- ✅ Groq statt Ollama (3x schneller, kein OOM)

**Kostenoptimierung:**
- ✅ Anthropic Prompt Caching (-90% System-Prompt-Kosten)
- ✅ Risk-Stratified Verifier (-60% Verifier-Calls)
- ✅ BM25 Fast-Path (-60% Embedding-Calls)
- ✅ Query Embedding Cache (-30% Remote Embeddings)
- ✅ Context Budgeting (-40% Premium Input)

**Sicherheit:**
- ✅ Global Kill Switch (Soft → Throttle → Kill)
- ✅ Idempotency Keys (keine Doppel-Calls)
- ✅ Patch Risk Score (statt fixes Line-Limit)

### v1.2 (Januar 2026) - Production-Ready

- ✅ Rate Limiting (Token-aware + Daily Budget)
- ✅ Adaptive Keepalive (traffic-basiert)
- ✅ Hybrid Retrieval (BM25 + Embeddings)
- ✅ Capability-based Tools
- ✅ Event-Driven Cache Invalidation
- ✅ Verifier mit JSON-Output

### v1.1 (Dezember 2025) - Security Hardening

- ✅ Hard Policy Gate VOR Router
- ✅ Circuit Breaker + Fallback-Kette
- ✅ Transaction-Pattern für Patches
- ✅ Haiku-Verifier für Semantic Cache

### v1.0 (November 2025) - Initial

- ✅ Dreistufiges Routing
- ✅ Zweistufiges Caching (Exact + Semantic)
- ✅ Working-Tree Fingerprint
- ✅ Ollama für lokales Routing

---

## Anhang A: Schnellstart-Checkliste

```
□ Hetzner CX22 bestellt (€4,35/Monat)
□ Ubuntu 24.04 installiert
□ SSH-Key eingerichtet
□ Firewall konfiguriert (22, 80, 443)
□ Python 3.11+ installiert
□ FastAPI + Dependencies installiert
□ Groq API-Key erhalten (groq.com)
□ Anthropic API-Key erhalten (anthropic.com)
□ .env mit Keys konfiguriert
□ Systemd Service eingerichtet
□ Nginx + SSL konfiguriert
□ /health Endpoint erreichbar
□ Erster Test-Request erfolgreich
□ Monitoring eingerichtet
□ Alerting konfiguriert
□ Backup-Script eingerichtet
```

---

## Anhang B: Troubleshooting

### Problem: Groq-Timeouts

```python
# Symptom: httpx.TimeoutException
# Lösung: Retry mit Backoff + Haiku-Fallback

@retry(stop=stop_after_attempt(3), wait=wait_exponential(min=0.1, max=2))
async def groq_classify_with_retry(query: str):
    ...
```

### Problem: Hohe Premium-Rate (>30%)

```python
# Diagnose:
# 1. Router-Logs prüfen: Welche Queries → Premium?
# 2. Häufige Patterns identifizieren
# 3. Router-Prompt anpassen

# Beispiel: "Erkläre mir X" sollte CHEAP sein, nicht PREMIUM
```

### Problem: Cache-Hit-Rate niedrig (<20%)

```python
# Mögliche Ursachen:
# 1. Fingerprint ändert sich zu oft → Git-Hooks prüfen
# 2. Similarity-Threshold zu hoch → 0.92 → 0.90
# 3. Wenig wiederholte Queries → Cache-Warmup
```

### Problem: Kill Switch zu früh aktiv

```python
# Anpassung der Limits in .env:
DAILY_BUDGET_SOFT=10.0   # Statt 5
DAILY_BUDGET_MEDIUM=30.0  # Statt 15
DAILY_BUDGET_HARD=100.0   # Statt 50
```

---

**Dokument-Version:** 1.3.0  
**Letzte Aktualisierung:** Februar 2026  
**Autor:** Sp4cerat
**Lizenz:** MIT
