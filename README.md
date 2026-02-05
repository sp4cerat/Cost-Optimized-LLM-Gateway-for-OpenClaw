
# LLM Gateway Implementation Concept v1.3

**Cost-Optimized AI Routing for OpenClaw**

Version 1.3 (Cost-Optimized) – February 2026

> **VPS + Groq Router + Prompt Caching = 73% Cost Reduction**
> 
> Monthly Costs: ~€25-30 instead of €92 (Cloud baseline)

---

## Table of Contents

1. [Executive Summary](#1-executive-summary)
2. [Architecture Overview](#2-architecture-overview)
3. [Infrastructure: VPS Instead of Cloud Platform](#3-infrastructure-vps-instead-of-cloud-platform)
4. [Three-Tier Routing](#4-three-tier-routing)
5. [Groq as Router](#5-groq-as-router)
6. [Hard Policy Gate (Security)](#6-hard-policy-gate-security)
7. [Rate Limiting & Kill Switch](#7-rate-limiting--kill-switch)
8. [Two-Stage Caching](#8-two-stage-caching)
9. [Hybrid Retrieval (BM25 + Embeddings)](#9-hybrid-retrieval-bm25--embeddings)
10. [Anthropic Prompt Caching](#10-anthropic-prompt-caching)
11. [Context Budgeting & Compression](#11-context-budgeting--compression)
12. [Capability-based Tools](#12-capability-based-tools)
13. [Deterministic Patching](#13-deterministic-patching)
14. [Patch Risk Score](#14-patch-risk-score)
15. [Risk-Stratified Verifier](#15-risk-stratified-verifier)
16. [Monitoring & KPIs](#16-monitoring--kpis)
17. [Implementation Plan](#17-implementation-plan)
18. [Cost Forecast](#18-cost-forecast)
19. [Changelog](#19-changelog)

---

## 1. Executive Summary

### 1.1 Problem

OpenClaw (AI Coding Assistant) requires LLM access for:
- Code explanations and documentation
- Bug fixes and refactoring suggestions
- Shell command generation
- Code reviews

**Challenge:** Claude Sonnet costs $3/1M Input + $15/1M Output. At 100+ requests/day, costs quickly exceed $100+/month.

### 1.2 Solution

An intelligent gateway with:

| Component | Function | Cost Impact |
|-----------|----------|-------------|
| **Three-Tier Routing** | Simple questions → cheap models | -60% API costs |
| **Two-Stage Caching** | Exact + Semantic Cache | -30% redundant calls |
| **Groq Router** | Fast intent classification | 3x faster than Ollama |
| **Prompt Caching** | Anthropic cached system prompts | -90% system prompt costs |
| **Context Budgeting** | Limited input tokens per tier | -40% premium input |
| **VPS Infrastructure** | Fixed price instead of pay-per-use | -€60/month |

### 1.3 Cost Comparison

| Scenario | Cloud Platform (v1.2) | VPS (v1.3) | Savings |
|----------|----------------------|------------|---------|
| Server | $74 (2 vCPU / 4GB) | €8.50 (4GB VPS) | -€60 |
| Storage | $5 (block storage) | €0 (included) | -€5 |
| Traffic | $5-10 | €0 (20TB included) | -€7 |
| Router | $0 (Ollama) | €1-2 (Groq) | +€1.50 |
| Premium (with caching) | $25-35 | €8-12 | -€20 |
| Embeddings + Verifier | $3-5 | €1-2 | -€3 |
| **TOTAL** | **~€92/month** | **~€25/month** | **-73%** |

### 1.4 Architecture Decision

| Option | When to Choose? | Cost |
|--------|----------------|------|
| **4GB VPS + Groq** | Standard recommendation; low cost, easy maintenance | €25-30/month |
| **8GB VPS + Ollama** | When 100% local/offline is required | €35/month |
| **Cloud Platform + Ollama** | Only when cloud ecosystem is mandatory (IAM, VPC) | €90+/month |

---

## 2. Architecture Overview

### 2.1 Request Flow

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                              REQUEST FLOW                                    │
└─────────────────────────────────────────────────────────────────────────────┘

  OpenClaw Request
        │
        ▼
┌───────────────────┐
│  Hard Policy Gate │ ──── BLOCK ───► 403 Forbidden
│  (Security)       │                 (rm -rf, secrets, etc.)
└───────────────────┘
        │ PASS
        ▼
┌───────────────────┐
│  Rate Limiter     │ ──── BLOCK ───► 429 Too Many Requests
│  + Budget Guard   │                 (Daily cap reached)
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
│  (Intent Class.)  │
└───────────────────┘
        │
        ├── CACHE_ONLY ──► "Please be more specific"
        │
        ├── LOCAL ──────► Haiku (cheap)
        │
        ├── CHEAP ──────► Haiku (cheap)
        │
        └── PREMIUM ────► Sonnet (with Prompt Caching)
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

### 2.2 Component Overview

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                        4GB VIRTUAL PRIVATE SERVER                            │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐        │
│  │   FastAPI   │  │   SQLite    │  │   SQLite    │  │   Nginx     │        │
│  │   Gateway   │  │ Exact Cache │  │Semantic Cache│ │   Reverse   │        │
│  │   (uvicorn) │  │   (FTS5)    │  │ + Embeddings │  │   Proxy     │        │
│  └─────────────┘  └─────────────┘  └─────────────┘  └─────────────┘        │
│                                                                             │
├─────────────────────────────────────────────────────────────────────────────┤
│                           EXTERNAL SERVICES                                  │
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

## 3. Infrastructure: VPS Instead of Cloud Platform

### 3.1 Why a VPS?

| Aspect | Cloud Instance (4GB burstable) | 4GB VPS (dedicated) | Advantage |
|--------|-------------------------------|---------------------|-----------|
| **Price** | $74/month + extras | ~€4.35/month fixed | -90% base |
| **RAM** | 4GB (burstable) | 4GB (dedicated) | No throttling |
| **Storage** | $0.10/GB (block) | 40GB included | No IOPS costs |
| **Traffic** | $0.09/GB outbound | 20TB included | No surprises |
| **Location** | Variable | EU datacenter | GDPR, low latency |
| **Complexity** | IAM, VPC, SG | SSH + Firewall | Simpler |

### 3.2 Server Options

| Server | RAM | vCPU | SSD | Price | Use Case |
|--------|-----|------|-----|-------|----------|
| **4GB VPS** | 4GB | 2 | 40GB | ~€4.35 | Groq router (recommended) |
| **8GB VPS** | 8GB | 4 | 80GB | ~€7.05 | Ollama local |
| **16GB VPS** | 16GB | 8 | 160GB | ~€14.76 | Heavy workload |

### 3.3 Server Setup

```bash
# 1. Order a 4GB VPS with Ubuntu 24.04 LTS from your provider

# 2. Set up SSH access
ssh root@<server-ip>

# 3. Base setup
apt update && apt upgrade -y
apt install -y python3-pip python3-venv nginx certbot python3-certbot-nginx

# 4. Configure firewall
ufw allow 22/tcp
ufw allow 80/tcp
ufw allow 443/tcp
ufw enable

# 5. Create non-root user
adduser gateway
usermod -aG sudo gateway
su - gateway

# 6. Install gateway
cd /opt
sudo mkdir llm-gateway && sudo chown gateway:gateway llm-gateway
cd llm-gateway
python3 -m venv venv
source venv/bin/activate

# 7. Install dependencies
pip install fastapi uvicorn httpx anthropic openai tenacity sqlite-utils

# 8. Environment variables
cat > .env << 'EOF'
GROQ_API_KEY=gsk_...
ANTHROPIC_API_KEY=sk-ant-...
OPENAI_API_KEY=sk-...  # Only for embeddings fallback
GATEWAY_SECRET=<random-secret>
DAILY_BUDGET_HARD=50.0
DAILY_BUDGET_SOFT=5.0
EOF

# 9. Systemd service
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

# 10. Nginx reverse proxy + SSL
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

# 11. SSL with Let's Encrypt
sudo certbot --nginx -d gateway.yourdomain.com
```

---

## 4. Three-Tier Routing

### 4.1 Tier Definitions

| Tier | Model | Cost | Latency | Use Case |
|------|-------|------|---------|----------|
| **LOCAL** | Haiku (via Groq Fallback) | $0.25/1M | 200ms | Simple questions |
| **CHEAP** | Claude Haiku | $0.25/1M in, $1.25/1M out | 300ms | Explanations, docs |
| **PREMIUM** | Claude Sonnet | $3/1M in, $15/1M out | 500ms | Code generation, patches |

### 4.2 Router Logic

```python
from enum import Enum
from pydantic import BaseModel

class RouterAction(str, Enum):
    CACHE_ONLY = "cache_only"    # Too vague, no LLM call
    LOCAL = "local"              # Simple questions
    CHEAP = "cheap"              # Explanations, documentation
    PREMIUM = "premium"          # Code generation, complex tasks

class RouterResult(BaseModel):
    action: RouterAction
    confidence: float           # 0.0 - 1.0
    response_type: str          # For cache TTL and verifier
    reason: str                 # For logging/debugging

# Response types for cache management
RESPONSE_TYPES = {
    "explanation_generic": {
        "description": "General explanations (What is X?)",
        "ttl_base": 7 * 24 * 3600,  # 7 days
        "tier": "cheap"
    },
    "explanation_contextual": {
        "description": "Project-specific explanations",
        "ttl_base": 24 * 3600,  # 24 hours
        "tier": "cheap"
    },
    "code_suggestion": {
        "description": "Code suggestions, patches",
        "ttl_base": 0,  # Invalidated on git commit
        "tier": "premium"
    },
    "code_review": {
        "description": "Code reviews, best practices",
        "ttl_base": 12 * 3600,  # 12 hours
        "tier": "premium"
    },
    "command_execution": {
        "description": "Shell commands, CLI",
        "ttl_base": 3600,  # 1 hour
        "tier": "premium"
    },
    "documentation": {
        "description": "API docs, README generation",
        "ttl_base": 24 * 3600,
        "tier": "cheap"
    }
}
```

### 4.3 Router System Prompt

```python
ROUTER_SYSTEM_PROMPT = """You are an intent classifier for coding requests.

Classify the request into EXACTLY ONE category:

CACHE_ONLY - Request is too vague/unclear for a meaningful answer
  Examples: "help", "?", "code", "fix it"

LOCAL - Trivial questions that don't require code analysis
  Examples: "What does HTTP 404 mean?", "What's the command for..."

CHEAP - Explanations, documentation, general best practices
  Examples: "Explain async/await to me", "What's the difference between..."

PREMIUM - Code generation, patches, complex analysis, project-specific
  Examples: "Write a function that...", "Fix the bug in...", "Refactor..."

Reply ONLY with JSON:
{"action": "...", "confidence": 0.0-1.0, "response_type": "...", "reason": "..."}

response_type must be one of:
- explanation_generic
- explanation_contextual  
- code_suggestion
- code_review
- command_execution
- documentation"""
```

---

## 5. Groq as Router

### 5.1 Why Groq Instead of Ollama?

| Aspect | Ollama (local) | Groq (API) |
|--------|----------------|------------|
| **Latency** | 500-800ms (warm), 2-4s (cold) | 150-250ms (constant) |
| **RAM Usage** | 2.5 GB (30% of 8GB) | 0 GB |
| **OOM Risk** | Yes (with cache + embeddings) | No |
| **Maintenance** | Updates, monitoring, tuning | None |
| **Cost** | €0 (but larger server needed) | ~€1-2/month |
| **Offline Capable** | Yes | No |

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
    """Classifies intent via Groq Llama 3.1 8B"""
    
    user_content = f"Context: {context}\n\nQuery: {query}" if context else query
    
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

### 5.3 Groq Cost Calculation

```python
# Assumptions:
# - 500 requests/day
# - 300 tokens per router call (system prompt + query + response)

monthly_requests = 500 * 30  # = 15,000
tokens_per_request = 300
monthly_tokens = monthly_requests * tokens_per_request  # = 4,500,000

# Groq Llama 3.1 8B Instant Pricing (as of Feb 2026)
input_price = 0.05 / 1_000_000   # $0.05 per 1M input
output_price = 0.08 / 1_000_000  # $0.08 per 1M output

# ~90% are input (system prompt + query), ~10% output
input_tokens = monthly_tokens * 0.9  # = 4,050,000
output_tokens = monthly_tokens * 0.1  # = 450,000

monthly_cost = (input_tokens * input_price) + (output_tokens * output_price)
# = $0.2025 + $0.036 = $0.24/month

# With safety buffer: ~€1-2/month
```

### 5.4 Fallback Chain

```python
async def route_with_resilience(query: str, context: str = "") -> RouterResult:
    """Router with fallback chain: Groq → Haiku → Default"""
    
    # Primary: Groq (fast, cheap)
    try:
        return await groq_classify(query, context)
    except (httpx.TimeoutException, httpx.HTTPStatusError) as e:
        log.warning(f"Groq failed: {e}, falling back to Haiku")
        metrics.increment("router_fallback", tags={"from": "groq", "to": "haiku"})
    
    # Secondary: Haiku (more expensive, but reliable)
    try:
        return await haiku_classify(query, context)
    except Exception as e:
        log.error(f"Haiku classifier failed: {e}, using default")
        metrics.increment("router_fallback", tags={"from": "haiku", "to": "default"})
    
    # Last resort: Safety mode → Premium
    return RouterResult(
        action=RouterAction.PREMIUM,
        confidence=0.5,
        response_type="code_suggestion",
        reason="fallback_default"
    )
```

---

## 6. Hard Policy Gate (Security)

### 6.1 Why Before the Router?

The Hard Policy Gate runs **BEFORE** the LLM router to:
1. Block dangerous requests immediately (no LLM call needed)
2. Intercept prompt injection attempts
3. Save costs for obviously illegitimate requests

### 6.2 Blocklists

```python
import re
from typing import Optional

class PolicyViolation(Exception):
    def __init__(self, category: str, pattern: str, message: str):
        self.category = category
        self.pattern = pattern
        self.message = message

# Dangerous shell commands
DANGEROUS_COMMANDS = [
    # Destructive operations
    r"\brm\s+(-[rf]+\s+)*(/|~|\$HOME|\*)",
    r"\bmkfs\b",
    r"\bdd\s+.*of=/dev/",
    r"\b:(){.*};\s*:",  # Fork bomb
    r"\bchmod\s+(-R\s+)?[0-7]*777",
    r"\bchown\s+-R\s+.*\s+/",
    
    # Network attacks
    r"\bnc\s+.*-e\s+/bin/(ba)?sh",
    r"\bcurl\s+.*\|\s*(ba)?sh",
    r"\bwget\s+.*-O\s*-\s*\|\s*(ba)?sh",
    
    # Crypto mining
    r"\b(xmrig|minerd|cgminer|bfgminer)\b",
    
    # Service manipulation
    r"\bsystemctl\s+(stop|disable|mask)\s+(ssh|sshd|ufw|iptables)",
    r"\bservice\s+\w+\s+(stop|disable)",
    
    # Indirect destructive ops (NEW in v1.2)
    r"\bfind\s+.*-delete",
    r"\btruncate\s+--size\s*0",
]

# Sensitive data patterns
SENSITIVE_PATTERNS = [
    r"(api[_-]?key|secret|password|token)\s*[=:]\s*['\"]?[\w-]{20,}",
    r"-----BEGIN\s+(RSA\s+)?PRIVATE\s+KEY-----",
    r"aws_secret_access_key\s*=",
    r"ghp_[a-zA-Z0-9]{36}",  # GitHub token
    r"sk-[a-zA-Z0-9]{48}",   # OpenAI key
    r"sk-ant-[a-zA-Z0-9-]{95}",  # Anthropic key
]

# Forbidden paths
FORBIDDEN_PATHS = [
    r"/etc/(passwd|shadow|sudoers)",
    r"/root/",
    r"~/.ssh/",
    r"\.env",
    r"\.(pem|key|crt)$",
]

# Unicode normalization (against bypass attempts)
def normalize_unicode(text: str) -> str:
    """Normalizes Unicode tricks like invisible characters"""
    import unicodedata
    # NFKC normalizes ① → 1, ᴿ → R, etc.
    text = unicodedata.normalize("NFKC", text)
    # Remove zero-width characters
    text = re.sub(r"[\u200b-\u200f\u2060\ufeff]", "", text)
    return text

def check_hard_policy(query: str) -> Optional[PolicyViolation]:
    """
    Checks query against hard policy rules.
    Returns None if OK, otherwise PolicyViolation.
    """
    # Normalize first
    normalized = normalize_unicode(query.lower())
    
    # Check dangerous commands
    for pattern in DANGEROUS_COMMANDS:
        if re.search(pattern, normalized, re.IGNORECASE):
            return PolicyViolation(
                category="dangerous_command",
                pattern=pattern,
                message="Dangerous command detected"
            )
    
    # Check sensitive data (also in original, not just normalized)
    for pattern in SENSITIVE_PATTERNS:
        if re.search(pattern, query, re.IGNORECASE):
            return PolicyViolation(
                category="sensitive_data",
                pattern=pattern,
                message="Sensitive data detected in request"
            )
    
    # Check forbidden paths
    for pattern in FORBIDDEN_PATHS:
        if re.search(pattern, normalized, re.IGNORECASE):
            return PolicyViolation(
                category="forbidden_path",
                pattern=pattern,
                message="Access to protected path"
            )
    
    return None
```

### 6.3 Command Allowlist (Capability-Based)

```python
# Instead of blocklist: Explicit allowlist for tool operations
ALLOWED_COMMANDS = {
    # Git (read-mostly)
    "git": ["status", "diff", "log", "show", "branch", "remote", "fetch"],
    
    # Package managers (read + install)
    "npm": ["list", "outdated", "audit", "install", "ci", "run", "test"],
    "pip": ["list", "show", "check", "install"],
    
    # Build tools
    "make": ["*"],  # Makefile-defined targets
    "cargo": ["build", "test", "check", "clippy", "fmt"],
    
    # Docker (read-mostly)
    "docker": ["ps", "images", "logs", "inspect", "stats"],
    "docker-compose": ["ps", "logs", "config"],
    
    # System info (read-only)
    "ls": ["*"],
    "cat": ["*"],  # With path restriction!
    "head": ["*"],
    "tail": ["*"],
    "grep": ["*"],
    "find": ["-name", "-type", "-mtime"],  # No -delete!
    "du": ["*"],
    "df": ["*"],
    "free": ["*"],
    "top": ["-bn1"],
    "ps": ["aux", "-ef"],
}

def is_command_allowed(command: str, args: list[str]) -> bool:
    """Checks if command is on the allowlist"""
    if command not in ALLOWED_COMMANDS:
        return False
    
    allowed_args = ALLOWED_COMMANDS[command]
    if "*" in allowed_args:
        return True
    
    # Check each argument
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
                "rpm": 100,      # Requests per minute
                "tpm": 50_000,   # Tokens per minute
                "daily_usd": 0,  # Unlimited (local)
                "burst": 10      # Burst buffer
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
        
        # Tracking per tier
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
        
        # Daily reset
        if now.date() != self.last_reset.date():
            self.daily_spend.clear()
            self.last_reset = now
        
        # Cleanup old timestamps (>1 minute)
        cutoff = now - timedelta(minutes=1)
        self.minute_requests[tier] = [
            ts for ts in self.minute_requests[tier] if ts > cutoff
        ]
        
        # RPM check
        if len(self.minute_requests[tier]) >= limits["rpm"]:
            # Allow burst?
            if len(self.minute_requests[tier]) < limits["rpm"] + limits["burst"]:
                return (True, "burst_allowed", 0)
            return (False, "rpm_exceeded", 60)
        
        # TPM check
        if self.minute_tokens[tier] + estimated_tokens > limits["tpm"]:
            return (False, "tpm_exceeded", 30)
        
        # Daily budget check (premium)
        if limits["daily_usd"] > 0:
            estimated_cost = self._estimate_cost(tier, estimated_tokens)
            if self.daily_spend[tier] + estimated_cost > limits["daily_usd"]:
                return (False, "daily_budget_exceeded", 0)
        
        return (True, "ok", 0)
    
    def record(self, tier: str, tokens_used: int, cost_usd: float):
        """Records successful request"""
        self.minute_requests[tier].append(datetime.now())
        self.minute_tokens[tier] += tokens_used
        self.daily_spend[tier] += cost_usd
    
    def _estimate_cost(self, tier: str, tokens: int) -> float:
        """Estimates cost based on tier"""
        prices = {
            "local": 0,
            "cheap": 0.25 / 1_000_000,   # Haiku input
            "premium": 3.00 / 1_000_000   # Sonnet input (output separate)
        }
        return tokens * prices.get(tier, 0)
```

### 7.2 Global Kill Switch (NEW in v1.3)

```python
class BudgetGuard:
    """
    Three-tier budget system:
    - Soft: Warning + Log
    - Medium: Throttle (5s delay)
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
        # Daily reset
        today = datetime.now().date()
        if today != self.last_reset:
            self.today_spend = 0.0
            self.premium_disabled = False
            self.last_reset = today
            log.info("Daily budget reset")
        
        projected = self.today_spend + estimated_cost
        
        # Kill switch active?
        if self.premium_disabled:
            if tier == "premium":
                return {
                    "allowed": False,
                    "delay": 0,
                    "reason": "kill_switch_active"
                }
            # Local/Cheap still allowed
            return {"allowed": True, "delay": 0, "reason": "non_premium_allowed"}
        
        # Hard limit → Activate kill switch
        if projected > self.limits["hard"]:
            self.premium_disabled = True
            log.critical(f"KILL SWITCH ACTIVATED: ${projected:.2f} > ${self.limits['hard']}")
            self._alert_admin(f"Budget Kill Switch: ${projected:.2f}")
            return {
                "allowed": False,
                "delay": 0,
                "reason": "hard_limit_kill_switch"
            }
        
        # Medium limit → Throttle
        if projected > self.limits["medium"]:
            log.error(f"Budget ${projected:.2f} > Medium ${self.limits['medium']}, THROTTLING")
            return {
                "allowed": True,
                "delay": 5.0,
                "reason": "throttle_medium_limit"
            }
        
        # Soft limit → Warning
        if projected > self.limits["soft"]:
            log.warning(f"Budget ${projected:.2f} > Soft ${self.limits['soft']}")
        
        return {"allowed": True, "delay": 0, "reason": "ok"}
    
    def record_spend(self, cost: float):
        """Records actual costs"""
        self.today_spend += cost
        metrics.gauge("daily_spend_usd", self.today_spend)
    
    def _alert_admin(self, message: str):
        """Sends alert (webhook, email, etc.)"""
        # TODO: Implement alerting
        pass
```

### 7.3 Idempotency Keys (Prevents Duplicate Calls)

```python
import hashlib
from datetime import datetime, timedelta

class IdempotencyGuard:
    """
    Prevents duplicate API calls during retries/timeouts.
    Caches response for identical requests for 5 minutes.
    """
    
    def __init__(self, ttl_seconds: int = 300):
        self.cache: dict[str, tuple[dict, datetime]] = {}
        self.ttl = timedelta(seconds=ttl_seconds)
    
    def get_key(self, request: dict) -> str:
        """Generates unique key from request content"""
        content = json.dumps({
            "model": request.get("model"),
            "messages": request.get("messages"),
            "temperature": request.get("temperature", 1),
            "max_tokens": request.get("max_tokens")
        }, sort_keys=True)
        return hashlib.sha256(content.encode()).hexdigest()[:16]
    
    def check(self, key: str) -> Optional[dict]:
        """Checks if request was already processed"""
        self._cleanup()
        
        if key in self.cache:
            response, timestamp = self.cache[key]
            log.info(f"Idempotency hit: {key}")
            metrics.increment("idempotency_hits")
            return response
        
        return None
    
    def store(self, key: str, response: dict):
        """Stores response for key"""
        self.cache[key] = (response, datetime.now())
    
    def _cleanup(self):
        """Removes expired entries"""
        now = datetime.now()
        expired = [k for k, (_, ts) in self.cache.items() if now - ts > self.ttl]
        for k in expired:
            del self.cache[k]
```

---

## 8. Two-Stage Caching

### 8.1 Cache Architecture

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
    Exact cache based on SHA-256 hash.
    Uses working-tree fingerprint for context sensitivity.
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
        """Generates cache key from query + fingerprint"""
        content = f"{query}|{fingerprint}"
        return hashlib.sha256(content.encode()).hexdigest()
    
    def get(self, query: str, fingerprint: str) -> Optional[dict]:
        """Looks up exact cache entry"""
        cache_key = self.get_key(query, fingerprint)
        now = datetime.now()
        
        row = self.db.execute("""
            SELECT response, response_type FROM exact_cache
            WHERE cache_key = ? AND expires_at > ?
        """, (cache_key, now)).fetchone()
        
        if row:
            # Update hit counter
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
        """Stores response in cache"""
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
        """Invalidates all entries with old fingerprint"""
        self.db.execute("""
            DELETE FROM exact_cache WHERE fingerprint = ?
        """, (old_fingerprint,))
        self.db.commit()
    
    def cleanup_expired(self):
        """Removes expired entries"""
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
    Creates fingerprint from git status + relevant files.
    Change in fingerprint → Cache miss (correct!).
    """
    components = []
    
    try:
        # 1. Git HEAD commit
        head = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=project_path,
            capture_output=True,
            text=True
        )
        if head.returncode == 0:
            components.append(f"head:{head.stdout.strip()[:12]}")
        
        # 2. Git diff (staged + unstaged)
        diff = subprocess.run(
            ["git", "diff", "--stat", "HEAD"],
            cwd=project_path,
            capture_output=True,
            text=True
        )
        if diff.stdout.strip():
            diff_hash = hashlib.sha256(diff.stdout.encode()).hexdigest()[:8]
            components.append(f"diff:{diff_hash}")
        
        # 3. Active files (currently being edited)
        # This info comes from the OpenClaw client
        active_files = get_active_files_from_context()
        if active_files:
            files_hash = hashlib.sha256(
                "|".join(sorted(active_files)).encode()
            ).hexdigest()[:8]
            components.append(f"active:{files_hash}")
        
        # 4. Package lock (dependency changes)
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
    Semantic cache with embedding similarity.
    Uses local BM25 + remote embeddings (fallback).
    """
    
    def __init__(self, db_path: str = "semantic_cache.sqlite", 
                 similarity_threshold: float = 0.92):
        self.db = Database(db_path)
        self.threshold = similarity_threshold
        
        # Table for embeddings
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
        """Searches for semantically similar cache entry"""
        
        # Get embedding for query (with cache)
        query_embedding = await self.get_embedding(query)
        
        # Load all entries with same fingerprint
        rows = list(self.db.execute("""
            SELECT id, query, query_embedding, response, response_type
            FROM semantic_cache 
            WHERE fingerprint = ?
        """, [fingerprint]).fetchall())
        
        if not rows:
            return None
        
        # Calculate similarity
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
        """Gets embedding with query cache"""
        # See section 9.3 for Query Embedding Cache
        return await query_embedding_cache.get_or_compute(text)

def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    """Computes cosine similarity between two vectors"""
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))
```

### 8.5 Event-Driven Cache Invalidation

```python
# Git hook: .git/hooks/post-commit
#!/bin/bash
# Invalidates code-related cache entries after commit

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
# Gateway endpoint for invalidation
@app.post("/cache/invalidate")
async def invalidate_cache(event: CacheInvalidationEvent, auth: AuthDep):
    """Invalidates cache based on events"""
    
    if event.event == "git_commit":
        # Invalidate all code_suggestion entries
        exact_cache.db.execute("""
            DELETE FROM exact_cache 
            WHERE response_type = 'code_suggestion'
        """)
        
        # Semantic cache: Only if affected files
        if event.files:
            semantic_cache.invalidate_by_files(event.files)
        
        log.info(f"Cache invalidated for commit {event.commit[:8]}")
        return {"invalidated": True, "reason": "git_commit"}
    
    elif event.event == "manual":
        # Full invalidation
        exact_cache.db.execute("DELETE FROM exact_cache")
        semantic_cache.db.execute("DELETE FROM semantic_cache")
        return {"invalidated": True, "reason": "manual_full"}
    
    return {"invalidated": False, "reason": "unknown_event"}
```

### 8.6 Adaptive TTLs

```python
def get_adaptive_ttl(response_type: str, hit_count: int = 0) -> int:
    """
    Calculates TTL based on response type and popularity.
    Popular entries live longer, unused ones are deleted faster.
    """
    base_ttls = {
        "explanation_generic": 7 * 24 * 3600,    # 7 days
        "explanation_contextual": 24 * 3600,      # 24 hours
        "code_suggestion": 0,                     # Invalid on commit
        "code_review": 12 * 3600,                 # 12 hours
        "command_execution": 3600,                # 1 hour
        "documentation": 24 * 3600,               # 24 hours
    }
    
    base = base_ttls.get(response_type, 3600)
    
    # Popularity multiplier
    if hit_count >= 10:
        return int(base * 2)    # Very popular → 2x TTL
    elif hit_count >= 5:
        return int(base * 1.5)  # Popular → 1.5x TTL
    elif hit_count == 0:
        return int(base * 0.5)  # Unused → 0.5x TTL
    
    return base
```

---

## 9. Hybrid Retrieval (BM25 + Embeddings)

### 9.1 Why Hybrid?

| Method | Strengths | Weaknesses |
|--------|-----------|------------|
| **BM25 (local)** | Exact terms, IDs, error codes | No semantics |
| **Embeddings (remote)** | Semantic similarity | Cost, latency, privacy |

**Solution:** BM25 as fast path for 80% of queries, remote embeddings only as fallback.

### 9.2 BM25 with SQLite FTS5

```python
class BM25Search:
    """Local full-text search with SQLite FTS5"""
    
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
        """BM25 search with relevance score"""
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
        """Prepares query for FTS5"""
        # Escape special characters
        query = re.sub(r'[^\w\s-]', ' ', query)
        # Tokenize and join with OR
        tokens = query.split()
        return " OR ".join(tokens)
```

### 9.3 Query Embedding Cache (NEW in v1.3)

```python
class QueryEmbeddingCache:
    """
    Query vectors are stable → cache for 30 days.
    Reduces remote embedding calls by 60-70%.
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
        """Gets embedding from cache or computes new one"""
        
        # Normalize query
        normalized = self._normalize(query)
        query_hash = hashlib.sha256(normalized.encode()).hexdigest()[:16]
        
        # Check cache
        row = self.db.execute("""
            SELECT embedding FROM query_embeddings WHERE query_hash = ?
        """, (query_hash,)).fetchone()
        
        if row:
            metrics.increment("query_embedding_cache_hit")
            return np.frombuffer(row[0], dtype=np.float32)
        
        # Compute + store
        embedding = await self._compute_embedding(query)
        self.db.execute("""
            INSERT OR REPLACE INTO query_embeddings (query_hash, query_normalized, embedding)
            VALUES (?, ?, ?)
        """, (query_hash, normalized, embedding.tobytes()))
        self.db.commit()
        
        metrics.increment("query_embedding_cache_miss")
        return embedding
    
    def _normalize(self, query: str) -> str:
        """Normalizes query for cache key"""
        # Lowercase, strip, collapse whitespace
        query = query.lower().strip()
        query = re.sub(r'\s+', ' ', query)
        # Remove variable IDs (UUIDs, SHAs, etc.)
        query = re.sub(r'[a-f0-9]{8,}', '<ID>', query)
        return query
    
    async def _compute_embedding(self, text: str) -> np.ndarray:
        """Computes embedding via OpenAI API"""
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
        """Removes entries older than X days"""
        self.db.execute("""
            DELETE FROM query_embeddings 
            WHERE created_at < datetime('now', ?)
        """, (f'-{days} days',))
        self.db.commit()

# Global instance
query_embedding_cache = QueryEmbeddingCache()
```

### 9.4 Fast-Path Detection (NEW in v1.3)

```python
# Patterns that BM25 matches well
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
    """Detects technical queries that BM25 finds well"""
    return any(re.search(p, query, re.IGNORECASE) for p in TECHNICAL_PATTERNS)

async def hybrid_retrieval(query: str, project_context: dict) -> list[dict]:
    """
    Hybrid retrieval: BM25 fast path + remote embeddings fallback.
    """
    
    # FAST PATH: Technical queries → BM25 only
    if is_technical_query(query):
        bm25_results = bm25_search.search(query, limit=5)
        
        if bm25_results and bm25_results[0]["score"] > 8.0:
            log.debug(f"Fast-path: BM25 hit for technical query")
            metrics.increment("retrieval_fast_path")
            return bm25_results
    
    # SLOW PATH: Semantic queries → Hybrid
    bm25_results = bm25_search.search(query, limit=10)
    
    # Remote embeddings only when BM25 is poor
    if not bm25_results or bm25_results[0]["score"] < 5.0:
        # Content filter before remote (privacy)
        if not contains_sensitive_content(query):
            embedding_results = await embedding_search(query, limit=5)
            return merge_and_rerank(bm25_results, embedding_results)
    
    return bm25_results

def contains_sensitive_content(text: str) -> bool:
    """Checks if text contains sensitive data → no remote embedding"""
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

### 10.1 How It Works

Anthropic automatically caches identical prompt prefixes. The system prompt is only processed on the first call, then loaded from cache.

**Prices:**
| Operation | Price (Sonnet) | vs. Normal |
|-----------|----------------|------------|
| Write to cache | $3.75/1M | +25% |
| Read from cache | $0.30/1M | **-90%** |
| Normal (no cache) | $3.00/1M | Baseline |

### 10.2 Implementation

```python
import anthropic

client = anthropic.Anthropic()

# Static system prompt (will be cached)
STATIC_SYSTEM_PROMPT = """You are a senior software engineer with expertise in:
- Web Development (React, Node.js, Python)
- DevOps (Docker, Kubernetes, CI/CD)
- Databases (PostgreSQL, MongoDB, Redis)

Your communication style:
- Precise and technically correct
- Code examples always as complete, runnable snippets
- For code changes: Unified Diff format
- When uncertain: Communicate explicitly

Output format for code changes:
```diff
--- a/path/to/file
+++ b/path/to/file
@@ -10,5 +10,7 @@
 context line
-old line
+new line
 context line
```

[... additional 2500 tokens of instructions ...]
"""

async def call_premium_with_caching(
    user_query: str,
    context: dict,
    max_tokens: int = 4096
) -> dict:
    """
    Premium call with prompt caching.
    System prompt is cached after first call (90% discount).
    """
    
    # Dynamic context in user message (NOT in system prompt!)
    user_message = f"""
Project context:
- Path: {context.get('project_path', 'N/A')}
- Framework: {context.get('framework', 'N/A')}
- Git status: {context.get('git_status', 'clean')}

Active files:
{context.get('active_files_summary', 'None')}

Query:
{user_query}
"""
    
    response = client.messages.create(
        model="claude-sonnet-4-20250514",
        max_tokens=max_tokens,
        system=[
            {
                "type": "text",
                "text": STATIC_SYSTEM_PROMPT,
                "cache_control": {"type": "ephemeral"}  # IMPORTANT!
            }
        ],
        messages=[
            {"role": "user", "content": user_message}
        ]
    )
    
    # Log cache status
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

### 10.3 Cost Calculation

```python
# Assumptions:
# - 30 premium requests/day
# - 3000 token system prompt
# - 1000 token user message (variable, not cached)
# - 2000 token output

# WITHOUT prompt caching:
system_cost = 3000 * (3.00 / 1_000_000)  # = $0.009 per request
user_cost = 1000 * (3.00 / 1_000_000)    # = $0.003 per request
output_cost = 2000 * (15.00 / 1_000_000) # = $0.030 per request
total_per_request = 0.009 + 0.003 + 0.030  # = $0.042

monthly_requests = 30 * 30  # = 900
monthly_cost_without_cache = 900 * 0.042  # = $37.80/month

# WITH prompt caching:
# First request: Cache write ($3.75/1M)
first_request_system = 3000 * (3.75 / 1_000_000)  # = $0.01125

# Subsequent requests: Cache read ($0.30/1M)
cached_system_cost = 3000 * (0.30 / 1_000_000)  # = $0.0009 per request

# Monthly costs
first_day_cost = 0.01125 + (29 * 0.0009) + (30 * (0.003 + 0.030))
# ≈ $1.02 (system) + $0.99 (user+output)

monthly_cost_with_cache = (
    0.01125 +                    # First cache write
    (899 * 0.0009) +             # Cache reads
    (900 * 0.003) +              # User messages
    (900 * 0.030)                # Outputs
)
# = $0.01 + $0.81 + $2.70 + $27.00 = $30.52/month

# Savings: $37.80 - $30.52 = $7.28/month (19%)
# With longer system prompt (6000 tokens): Savings double!
```

### 10.4 Best Practices

```python
# ✅ CORRECT: Static system prompt
SYSTEM_PROMPT = "You are an assistant for web development..."

# ❌ WRONG: Dynamic content in system prompt (cache miss!)
SYSTEM_PROMPT = f"Today is {date.today()}. You are an assistant..."

# ✅ CORRECT: Dynamic content in user message
user_message = f"Context: Today is {date.today()}\n\nQuestion: {query}"

# ✅ CORRECT: Long, stable instructions in system prompt
# ❌ WRONG: Short system prompts (little cache benefit)
```

---

## 11. Context Budgeting & Compression

### 11.1 Token Budget per Tier (NEW in v1.3)

| Tier | Max Input | Max Output | On Exceeding |
|------|-----------|------------|--------------|
| **Local** | 4,000 | 2,000 | Truncate + Warning |
| **Cheap** | 8,000 | 4,000 | Truncate + Warning |
| **Premium** | 16,000 | 8,000 | Compress + Summarize |

### 11.2 Context Budget Implementation

```python
import tiktoken

# Token encoder for Claude (approximated with cl100k_base)
encoder = tiktoken.get_encoding("cl100k_base")

def estimate_tokens(text: str) -> int:
    """Estimates token count"""
    return len(encoder.encode(text))

class ContextBudget:
    """Limits context size per tier"""
    
    LIMITS = {
        "local":   {"input": 4000,  "output": 2000},
        "cheap":   {"input": 8000,  "output": 4000},
        "premium": {"input": 16000, "output": 8000},
    }
    
    def apply(self, tier: str, context: dict) -> dict:
        """Applies budget to context"""
        limits = self.LIMITS[tier]
        
        total_tokens = 0
        budgeted_context = {}
        
        # 1. Query always complete
        query = context.get("query", "")
        query_tokens = estimate_tokens(query)
        total_tokens += query_tokens
        budgeted_context["query"] = query
        
        # 2. System prompt (cached for premium, still counts)
        system_tokens = estimate_tokens(context.get("system_prompt", ""))
        total_tokens += system_tokens
        
        remaining = limits["input"] - total_tokens
        
        # 3. Files by relevance
        files = context.get("files", [])
        budgeted_files = []
        
        for file in sorted(files, key=lambda f: f.get("relevance", 0), reverse=True):
            file_tokens = estimate_tokens(file.get("content", ""))
            
            if file_tokens <= remaining:
                budgeted_files.append(file)
                remaining -= file_tokens
            elif remaining > 500:
                # Excerpt instead of full file
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
        
        # 4. Compress logs
        if "logs" in context:
            compressed_logs = self._compress_logs(
                context["logs"], 
                max_tokens=min(remaining, 2000)
            )
            budgeted_context["logs"] = compressed_logs
        
        return budgeted_context
    
    def _extract_relevant(self, file: dict, query: str) -> str:
        """Extracts relevant code sections"""
        content = file.get("content", "")
        
        # Find relevant functions/classes
        # (simplified implementation)
        lines = content.split("\n")
        relevant_lines = []
        
        for i, line in enumerate(lines):
            # Search for query keywords
            if any(kw.lower() in line.lower() for kw in query.split()):
                # Context: 5 lines before/after
                start = max(0, i - 5)
                end = min(len(lines), i + 6)
                relevant_lines.extend(lines[start:end])
                relevant_lines.append("...")
        
        if relevant_lines:
            return "\n".join(relevant_lines[:100])  # Max 100 lines
        
        # Fallback: beginning of file
        return "\n".join(lines[:50])
    
    def _compress_logs(self, logs: str, max_tokens: int) -> str:
        """Compresses logs for context"""
        return compress_logs_for_context(logs, max_tokens)
```

### 11.3 Log/Trace Compression Pipeline

```python
import re

def compress_logs_for_context(logs: str, max_tokens: int = 2000) -> str:
    """
    Compresses logs through:
    1. Duplicate removal
    2. Timestamp normalization
    3. Stack trace shortening
    4. Token budget trimming
    """
    lines = logs.split("\n")
    
    # 1. Remove duplicates (common in retry loops)
    seen = set()
    unique_lines = []
    
    for line in lines:
        # Normalize: Remove timestamps, IDs
        normalized = re.sub(
            r'\d{4}-\d{2}-\d{2}[T ]\d{2}:\d{2}:\d{2}[.\d]*Z?',
            '<TIME>',
            line
        )
        normalized = re.sub(r'[a-f0-9]{8,}', '<ID>', normalized)
        
        if normalized not in seen:
            seen.add(normalized)
            unique_lines.append(line)
    
    # 2. Shorten stack traces (only first + last 3 frames)
    compressed = compress_stacktraces(unique_lines)
    
    # 3. Trim to token budget
    result = "\n".join(compressed)
    while estimate_tokens(result) > max_tokens and compressed:
        compressed = compressed[1:]  # Remove oldest line
        result = "\n".join(compressed)
    
    return result

def compress_stacktraces(lines: list[str]) -> list[str]:
    """Shortens stack traces to essential frames"""
    result = []
    in_stacktrace = False
    stack_buffer = []
    
    for line in lines:
        # Detection: Stack trace start
        if re.match(r'\s*(at |File "|Traceback)', line):
            in_stacktrace = True
            stack_buffer.append(line)
            continue
        
        # End of stack trace
        if in_stacktrace and not re.match(r'\s*(at |File "|\s+\^)', line):
            in_stacktrace = False
            
            # Only first 3 + last 3 frames
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

### 12.1 Three-Zone Model

| Zone | Paths | AI Autonomy | Confirm? |
|------|-------|-------------|----------|
| **App/Project** | /srv/projects/*, dist, .cache | ✅ Full | No |
| **User/Runtime** | $HOME/.local, venv, docker | ⚠️ Restricted | On write |
| **System** | /etc, /usr, /var/lib, systemd | ❌ Blocked | Always |

### 12.2 Tool Definitions

```python
from pydantic import BaseModel, Field
from typing import Literal, Optional
from pathlib import Path

# Defined roots for different operations
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
    """Input for fs_delete tool"""
    path: str = Field(..., description="Path to delete")
    recursive: bool = Field(False, description="Delete recursively")
    dry_run: bool = Field(True, description="Simulate only")

class FSDeleteResult(BaseModel):
    """Result of fs_delete"""
    success: bool
    deleted_count: int
    total_size_bytes: int
    sample_files: list[str]
    dry_run: bool
    error: Optional[str] = None

async def fs_delete(input: FSDeleteInput) -> FSDeleteResult:
    """
    Safe deletion with trash retention.
    Only in allowed paths, with audit log.
    """
    path = Path(input.path).resolve()
    
    # 1. Check against forbidden paths
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
    
    # 2. Check against deletable roots
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
    
    # 3. Collect files
    if input.recursive and path.is_dir():
        files = list(path.rglob("*"))
    elif path.is_file():
        files = [path]
    else:
        files = list(path.glob("*"))
    
    total_size = sum(f.stat().st_size for f in files if f.is_file())
    sample = [str(f) for f in files[:5]]
    
    # 4. Dry run or actual deletion
    if input.dry_run:
        return FSDeleteResult(
            success=True,
            deleted_count=len(files),
            total_size_bytes=total_size,
            sample_files=sample,
            dry_run=True
        )
    
    # 5. Move to trash (7-day retention)
    trash_path = Path(f"/srv/trash/{datetime.now().strftime('%Y%m%d')}")
    trash_path.mkdir(parents=True, exist_ok=True)
    
    import shutil
    shutil.move(str(path), str(trash_path / path.name))
    
    # 6. Audit log
    log.info(f"fs_delete: {path} → {trash_path}, {len(files)} files, {total_size} bytes")
    
    return FSDeleteResult(
        success=True,
        deleted_count=len(files),
        total_size_bytes=total_size,
        sample_files=sample,
        dry_run=False
    )
```

### 12.3 Cleanup Policies

```python
CLEANUP_POLICIES = {
    "node_cache_v1": {
        "description": "Clean Node.js build cache",
        "targets": [
            "node_modules/.cache/**",
            ".next/cache/**",
            "dist/**/*.map",
        ],
        "max_size_mb": 5000,
        "min_age_days": 0,
    },
    "build_artifacts_v1": {
        "description": "Build artifacts older than 7 days",
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
        "description": "Docker build cache and dangling images",
        "commands": [
            "docker image prune -f",
            "docker builder prune -f --filter until=168h",
        ],
        "max_size_mb": 20000,
        "requires_confirm": True,  # Always requires user confirmation!
    },
}

async def cleanup_project(project_id: str, policy_name: str, dry_run: bool = True):
    """Executes cleanup policy"""
    
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
        "details": results[:20]  # Only first 20 for overview
    }
```

---

## 13. Deterministic Patching

### 13.1 Why Deterministic?

LLMs are non-deterministic → the same request can produce different patches.
**Problem:** When re-executing, the patch may no longer match the code.

**Solution:**
1. **Fingerprint** captures the exact code state
2. **Verifier** checks if patch still applies
3. **Transaction pattern** makes patches atomic

### 13.2 Patch Format: Unified Diff

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
    """Parses unified diff into structured format"""
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
            old_content="",  # Loaded later
            new_content="",
            hunks=hunks
        ))
    
    return Patch(files=files, description="")
```

### 13.3 Transaction Pattern

```python
import shutil
from contextlib import contextmanager

@contextmanager
def patch_transaction(project_path: str):
    """
    Atomic patch application with rollback.
    On error: All changes are reverted.
    """
    backup_path = f"{project_path}/.patch-backup-{datetime.now().timestamp()}"
    changed_files = []
    
    try:
        yield PatchContext(project_path, backup_path, changed_files)
        
        # Commit: Delete backup
        if os.path.exists(backup_path):
            shutil.rmtree(backup_path)
        
        log.info(f"Patch committed: {len(changed_files)} files changed")
        
    except Exception as e:
        # Rollback: Restore all files from backup
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
        """Modifies file with automatic backup"""
        full_path = os.path.join(self.project_path, relative_path)
        backup_file = os.path.join(self.backup_path, relative_path)
        
        # Create backup
        os.makedirs(os.path.dirname(backup_file), exist_ok=True)
        if os.path.exists(full_path):
            shutil.copy2(full_path, backup_file)
        
        # Modify file
        with open(full_path, 'w') as f:
            f.write(new_content)
        
        self.changed_files.append(full_path)

# Usage
async def apply_patch_safely(project_path: str, patch: Patch):
    """Applies patch with transaction pattern"""
    
    with patch_transaction(project_path) as ctx:
        for file_change in patch.files:
            # Load current content
            full_path = os.path.join(project_path, file_change.path)
            
            if os.path.exists(full_path):
                with open(full_path) as f:
                    current_content = f.read()
            else:
                current_content = ""
            
            # Apply hunks
            new_content = apply_hunks(current_content, file_change.hunks)
            
            # Write with backup
            ctx.modify_file(file_change.path, new_content)
    
    return {"success": True, "files_changed": len(patch.files)}
```

---

## 14. Patch Risk Score

### 14.1 Why Risk Score Instead of Fixed Limits?

v1.2 used a fixed maximum (300 lines). Problem:
- Blocks legitimate refactorings
- Allows dangerous small changes (e.g., in auth/)

**v1.3 Solution:** Weighted risk score based on:
- Path risk (auth/, payment/, etc.)
- File type risk (.sql, .env, etc.)
- Change ratio (% of file changed)

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
    Calculates risk score for a patch.
    
    Score interpretation:
    - 0-7:   Low Risk → Auto-apply allowed
    - 8-14:  Medium Risk → Apply with review flag
    - 15+:   High Risk → Blocked
    """
    score = 0
    reasons = []
    
    for file in diff.files:
        path_lower = file.path.lower()
        
        # Path risk
        for pattern, weight in HIGH_RISK_PATHS.items():
            if pattern in path_lower:
                score += weight
                reasons.append(f"high_risk_path:{pattern}")
        
        # Extension risk
        for ext, weight in HIGH_RISK_EXTENSIONS.items():
            if path_lower.endswith(ext) or ext in path_lower:
                score += weight
                reasons.append(f"high_risk_extension:{ext}")
        
        # Change ratio risk
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
    
    # Number of files
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
    Validates patch against hard limits.
    Returns: (allowed, reason)
    """
    # Absolutely forbidden paths
    forbidden = ["/etc/", "/.env", "/.ssh/", "*.pem", "*.key", "id_rsa"]
    
    for file in diff.files:
        for pattern in forbidden:
            if pattern.startswith("*"):
                if file.path.endswith(pattern[1:]):
                    return False, f"Forbidden extension: {pattern}"
            elif pattern in file.path:
                return False, f"Forbidden path: {pattern}"
    
    # Check risk score
    risk = calculate_patch_risk(diff)
    if not risk.allowed:
        return False, f"Risk score too high: {risk.score} ({', '.join(risk.reasons)})"
    
    return True, "ok"
```

---

## 15. Risk-Stratified Verifier

### 15.1 Why Not Always Verify?

v1.2 called the Haiku verifier on **every** semantic cache hit.
**Problem:**
- Cost (even if small: $0.0003/call)
- Latency (+300-500ms)
- Unnecessary for low-risk response types

**v1.3 Solution:** Verifier only when there's actual risk.

### 15.2 Verifier Decision Tree

```python
def should_verify_cache_hit(
    response_type: str,
    similarity: float,
    fingerprint_changed: bool,
    file_touched: bool = False
) -> tuple[bool, str]:
    """
    Decides whether verifier is needed.
    
    Returns: (should_verify, reason)
    """
    
    # HIGH RISK: Always verify
    HIGH_RISK_TYPES = ["code_patch", "command_execution", "code_suggestion"]
    if response_type in HIGH_RISK_TYPES:
        return True, "high_risk_response_type"
    
    # Fingerprint unchanged → Very safe
    if not fingerprint_changed:
        return False, "fingerprint_unchanged"
    
    # MEDIUM RISK: Only with low similarity or file change
    MEDIUM_RISK_TYPES = ["explanation_contextual", "code_review"]
    if response_type in MEDIUM_RISK_TYPES:
        if similarity < 0.97:
            return True, "contextual_low_similarity"
        if file_touched:
            return True, "contextual_file_changed"
        return False, "contextual_high_similarity"
    
    # LOW RISK: Never verify
    LOW_RISK_TYPES = ["explanation_generic", "documentation"]
    if response_type in LOW_RISK_TYPES:
        return False, "generic_always_safe"
    
    # Unknown → Verify as safety measure
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
    Verifies whether cached response is still valid.
    
    Returns: {
        "verdict": "VALID" | "INVALID",
        "reason": str,
        "confidence": float,
        "suggestion": Optional[str]
    }
    """
    
    # Quick check: Fingerprint identical → Always valid
    if current_fingerprint == original_fingerprint:
        return {
            "verdict": "VALID",
            "reason": "fingerprint_unchanged",
            "confidence": 1.0
        }
    
    # Haiku verifier for complex checks
    verification_prompt = f"""Check whether this response is still correct:

ORIGINAL QUERY:
{query}

CACHED RESPONSE:
{json.dumps(cached_response.get('content', ''), indent=2)[:2000]}

CONTEXT CHANGES:
- Old fingerprint: {original_fingerprint}
- New fingerprint: {current_fingerprint}

Reply ONLY with JSON:
{{
    "verdict": "VALID" or "INVALID",
    "reason": "context_unchanged|version_mismatch|missing_files|code_changed|security_risk",
    "confidence": 0.0-1.0,
    "suggestion": "Optional: What should be regenerated"
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
        # Fallback on parse error
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
    Two-stage verification:
    1. Local heuristics (fast, free)
    2. LLM verifier (only when needed)
    """
    
    # STAGE 1: Local heuristics
    
    # 1a. Fingerprint identical → VALID
    if fingerprint_current == fingerprint_cached:
        return {"verdict": "VALID", "reason": "fingerprint_match", "stage": 1}
    
    # 1b. Only timestamp diff → probably VALID
    fp_current = parse_fingerprint(fingerprint_current)
    fp_cached = parse_fingerprint(fingerprint_cached)
    
    if fp_current.get("head") == fp_cached.get("head"):
        if fp_current.get("diff") == fp_cached.get("diff"):
            return {"verdict": "VALID", "reason": "same_head_and_diff", "stage": 1}
    
    # 1c. Check response type
    if cached.get("response_type") == "explanation_generic":
        return {"verdict": "VALID", "reason": "generic_always_valid", "stage": 1}
    
    # STAGE 2: LLM verifier (only for uncertain cases)
    return await verify_cache_hit(cached, fingerprint_current, fingerprint_cached, cached.get("query", ""))
```

---

## 16. Monitoring & KPIs

### 16.1 Complete KPI Table

| Metric | Description | Target | Warning | Critical | Action |
|--------|-------------|--------|---------|----------|--------|
| `daily_cost_usd` | Daily API costs | <$2 | >$5 | >$15 | Throttle/Kill |
| `groq_latency_p95_ms` | Router latency (95th percentile) | <300 | >500 | >1000 | Fallback Haiku |
| `cache_hit_rate_total` | Total cache hit rate | >40% | <25% | <15% | Cache config |
| `exact_cache_hit_rate` | Exact cache hit rate | >25% | <10% | <5% | Key format |
| `semantic_cache_hit_rate` | Semantic cache hit rate | >15% | <5% | <2% | Threshold |
| `premium_ratio` | Share of premium requests | <25% | >35% | >50% | Tune router |
| `verifier_skip_rate` | Verifier skips (risk-based) | >50% | <30% | N/A | Risk config |
| `verifier_reject_rate` | Verifier rejections | <20% | >40% | >60% | Threshold |
| `context_compression_ratio` | Context compression | >30% | <15% | N/A | Pipeline |
| `idempotency_hit_rate` | Duplicate requests | <5% | >10% | >20% | Client bug? |
| `patch_success_rate` | Successful patches | >90% | <75% | <50% | Diff format |
| `patch_high_risk_rate` | High-risk patches | <10% | >20% | >30% | Review |
| `policy_block_rate` | Hard policy blocks | Log | >5% | >10% | Injection? |
| `rate_limit_hits` | Rate limit hits | <5% | >15% | >25% | Limits |
| `cache_size_mb` | Cache size | <500 | >800 | >1000 | Eviction |
| `query_embedding_cache_hit` | Query embedding cache | >60% | <40% | <20% | TTL |

### 16.2 Alerting Rules

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

## 17. Implementation Plan

### 17.1 Phase 1: Infrastructure + Basics (Days 1-3)

| Day | Task | Deliverable | Go/No-Go |
|-----|------|-------------|----------|
| 1 | Order 4GB VPS, Ubuntu setup, firewall, SSH | Server reachable | SSH works |
| 2 | FastAPI skeleton, Groq router, Hard Policy Gate | `/health` + `/route` endpoints | Routing works |
| 3 | Rate limiting, kill switch, idempotency | Budget system active | Cap at $5 triggers |

**Day 1 Commands:**
```bash
# Order a 4GB VPS with Ubuntu 24.04 from your provider
# DNS: gateway.yourdomain.com → Server IP

ssh root@<ip>
apt update && apt upgrade -y
apt install -y python3-pip python3-venv nginx certbot python3-certbot-nginx
ufw allow 22,80,443/tcp && ufw enable
```

**Day 2 Skeleton:**
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

### 17.2 Phase 2: Caching + Premium (Days 4-7)

| Day | Task | Deliverable | Go/No-Go |
|-----|------|-------------|----------|
| 4 | BM25 (FTS5), query embedding cache | Retrieval works | Fast path active |
| 5 | Activate Anthropic prompt caching | Cache headers in logs | Cache read >0 |
| 6 | Semantic cache + risk-stratified verifier | Verifier skips in logs | Skip rate >50% |
| 7 | Context budgeting + log compression | Compression in logs | Ratio >30% |

**Day 5: Activate Prompt Caching**
```python
# Only this change needed:
response = client.messages.create(
    model="claude-sonnet-4-20250514",
    system=[{
        "type": "text",
        "text": STATIC_SYSTEM_PROMPT,
        "cache_control": {"type": "ephemeral"}  # <-- This line!
    }],
    messages=[...]
)

# Logging to verify:
usage = response.usage
log.info(f"Cache: read={usage.cache_read_input_tokens}, write={usage.cache_creation_input_tokens}")
```

### 17.3 Phase 3: Tools + Hardening (Days 8-10)

| Day | Task | Deliverable | Go/No-Go |
|-----|------|-------------|----------|
| 8 | Capability tools, patch risk score | Tools work | High-risk blocked |
| 9 | Monitoring dashboard, alerting | Grafana/Prometheus | Alerts work |
| 10 | 24h load test, cost audit | Test report | <€1 for test day |

**Day 10: Load Test**
```bash
# Simulate 500 requests over 24h
for i in {1..500}; do
  curl -X POST https://gateway.yourdomain.com/v1/chat/completions \
    -H "Authorization: Bearer $TOKEN" \
    -d '{"messages":[{"role":"user","content":"Explain async/await in JavaScript"}]}'
  sleep 172  # ~500 requests in 24h
done

# Expected costs:
# - 400 cache hits → $0
# - 80 cheap (Haiku) → $0.02
# - 20 premium (with caching) → $0.60
# TOTAL: ~$0.62 for 500 requests
```

### 17.4 Phase 4: Beta + Production (Days 11-14)

| Day | Task | Deliverable |
|-----|------|-------------|
| 11 | Beta rollout (10% traffic) | Monitoring active |
| 12 | Collect feedback, fine-tuning | Adjustments documented |
| 13 | Rollout to 50%, then 100% | Production traffic |
| 14 | Documentation, runbook | Operations guide |

---

## 18. Cost Forecast

### 18.1 Detailed Breakdown

| Component | Calculation | Monthly |
|-----------|------------|---------|
| **4GB VPS** | ~€4.35 fixed | €4.35 |
| **Groq Router** | 15,000 req × 300 tok × $0.05/1M | €0.25 |
| **Embeddings** | 5,000 queries × $0.02/1K tok | €1.00 |
| **Haiku (Cheap)** | 10,000 req × 2K tok × $0.25/1M | €5.00 |
| **Haiku (Verifier)** | 2,000 calls × 500 tok × $0.25/1M | €0.25 |
| **Sonnet (Premium)** | 3,000 req × 4K tok × $3/1M (input) | €3.60 |
| **Sonnet (Premium)** | 3,000 req × 2K tok × $15/1M (output) | €9.00 |
| **Prompt Cache Discount** | 3,000 req × 3K tok × ($3-$0.30)/1M | -€2.43 |
| **TOTAL** | | **€21.02** |

### 18.2 Scenario Comparison

| Scenario | Requests/Day | Premium % | Cache Hit % | Cost/Month |
|----------|-------------|-----------|-------------|------------|
| **Light** | 100 | 15% | 50% | €12-15 |
| **Standard** | 500 | 20% | 40% | €20-25 |
| **Heavy** | 1000 | 25% | 35% | €40-50 |
| **Worst Case** | 2000 | 40% | 20% | €80-100 |

### 18.3 Break-Even Analysis

```
Without gateway (direct Sonnet):
- 500 req/day × 6K tok × $18/1M = $162/month

With gateway:
- ~$25/month

Savings: $137/month = 85%
Break-even: From 50 requests/day
```

---

## 19. Changelog

### v1.3 (February 2026) - Cost-Optimized

**Infrastructure:**
- ✅ VPS instead of cloud platform (-€60/month)
- ✅ Groq instead of Ollama (3x faster, no OOM)

**Cost Optimization:**
- ✅ Anthropic Prompt Caching (-90% system prompt costs)
- ✅ Risk-Stratified Verifier (-60% verifier calls)
- ✅ BM25 Fast Path (-60% embedding calls)
- ✅ Query Embedding Cache (-30% remote embeddings)
- ✅ Context Budgeting (-40% premium input)

**Security:**
- ✅ Global Kill Switch (Soft → Throttle → Kill)
- ✅ Idempotency Keys (no duplicate calls)
- ✅ Patch Risk Score (instead of fixed line limit)

### v1.2 (January 2026) - Production-Ready

- ✅ Rate Limiting (token-aware + daily budget)
- ✅ Adaptive Keepalive (traffic-based)
- ✅ Hybrid Retrieval (BM25 + Embeddings)
- ✅ Capability-based Tools
- ✅ Event-Driven Cache Invalidation
- ✅ Verifier with JSON Output

### v1.1 (December 2025) - Security Hardening

- ✅ Hard Policy Gate BEFORE Router
- ✅ Circuit Breaker + Fallback Chain
- ✅ Transaction Pattern for Patches
- ✅ Haiku Verifier for Semantic Cache

### v1.0 (November 2025) - Initial

- ✅ Three-Tier Routing
- ✅ Two-Stage Caching (Exact + Semantic)
- ✅ Working-Tree Fingerprint
- ✅ Ollama for Local Routing

---

## Appendix A: Quick Start Checklist

```
□ 4GB VPS ordered (~€4.35/month)
□ Ubuntu 24.04 installed
□ SSH key configured
□ Firewall configured (22, 80, 443)
□ Python 3.11+ installed
□ FastAPI + dependencies installed
□ Groq API key obtained (groq.com)
□ Anthropic API key obtained (anthropic.com)
□ .env configured with keys
□ Systemd service set up
□ Nginx + SSL configured
□ /health endpoint reachable
□ First test request successful
□ Monitoring set up
□ Alerting configured
□ Backup script set up
```

---

## Appendix B: Troubleshooting

### Problem: Groq Timeouts

```python
# Symptom: httpx.TimeoutException
# Solution: Retry with backoff + Haiku fallback

@retry(stop=stop_after_attempt(3), wait=wait_exponential(min=0.1, max=2))
async def groq_classify_with_retry(query: str):
    ...
```

### Problem: High Premium Rate (>30%)

```python
# Diagnosis:
# 1. Check router logs: Which queries → Premium?
# 2. Identify frequent patterns
# 3. Adjust router prompt

# Example: "Explain X to me" should be CHEAP, not PREMIUM
```

### Problem: Low Cache Hit Rate (<20%)

```python
# Possible causes:
# 1. Fingerprint changes too often → Check git hooks
# 2. Similarity threshold too high → 0.92 → 0.90
# 3. Few repeated queries → Cache warmup
```

### Problem: Kill Switch Triggers Too Early

```python
# Adjust limits in .env:
DAILY_BUDGET_SOFT=10.0   # Instead of 5
DAILY_BUDGET_MEDIUM=30.0  # Instead of 15
DAILY_BUDGET_HARD=100.0   # Instead of 50
```

---

**Document Version:** 1.3.0  
**Last Updated:** February 2026  
**Author:** OpenClaw Team  
**License:** MIT
