"""FacePass server: enrollment website, scanner page, and the matching API."""
import base64
import hmac
import math
import os
import uuid

from flask import Flask, Response, abort, jsonify, request, send_from_directory

import db

DESCRIPTOR_LEN = 128
# face-api.js euclidean distance: < 0.6 is usually the same person. Lower = stricter.
MATCH_THRESHOLD = float(os.environ.get("MATCH_THRESHOLD", "0.5"))
ADMIN_KEY = os.environ.get("ADMIN_KEY", "")
MAX_PHOTO_BYTES = 400_000

app = Flask(__name__, static_folder="static", static_url_path="/static")
app.config["MAX_CONTENT_LENGTH"] = 2 * 1024 * 1024
db.init()


# ---------- pages ----------

@app.get("/")
def enroll_page():
    return send_from_directory("templates", "enroll.html")


@app.get("/scan")
def scan_page():
    return send_from_directory("templates", "scan.html")


@app.get("/admin")
def admin_page():
    return send_from_directory("templates", "admin.html")


@app.get("/healthz")
def health():
    return {"ok": True}


@app.after_request
def headers(resp):
    # Pages and API responses are never cached; static files are re-validated (ETag) on every load,
    # so phones and the app always pick up a new deploy.
    if request.path.startswith("/api/") or not request.path.startswith("/static/"):
        resp.headers["Cache-Control"] = "no-store"
    else:
        resp.headers["Cache-Control"] = "no-cache"
    return resp


# ---------- helpers ----------

def _clean_descriptor(d):
    if not isinstance(d, list) or len(d) != DESCRIPTOR_LEN:
        return None
    try:
        out = [float(x) for x in d]
    except (TypeError, ValueError):
        return None
    if not all(math.isfinite(x) for x in out):
        return None
    return out


def _clean_descriptors(raw, max_count):
    if not isinstance(raw, list) or not raw:
        return None
    out = [_clean_descriptor(d) for d in raw[:max_count]]
    return None if any(d is None for d in out) else out


def _dist(a, b):
    return math.sqrt(sum((x - y) ** 2 for x, y in zip(a, b)))


def _best_match(probes):
    """For each enrolled person, average (over probe frames) the distance to their closest stored sample."""
    best = None
    for pid, name, stored in db.all_descriptors():
        score = sum(min(_dist(p, s) for s in stored) for p in probes) / len(probes)
        if best is None or score < best[2]:
            best = (pid, name, score)
    return best


def _require_admin():
    if not ADMIN_KEY:
        abort(Response('{"error":"ADMIN_KEY is not set on the server"}', 503, mimetype="application/json"))
    given = request.headers.get("X-Admin-Key", "")
    if not hmac.compare_digest(given, ADMIN_KEY):
        abort(Response('{"error":"Wrong admin key"}', 401, mimetype="application/json"))


# ---------- API ----------

@app.post("/api/enroll")
def enroll():
    body = request.get_json(silent=True) or {}
    name = str(body.get("name", "")).strip()
    if not (1 <= len(name) <= 60):
        return jsonify(error="Please enter your name (1-60 characters)."), 400

    descriptors = _clean_descriptors(body.get("descriptors"), max_count=5)
    if not descriptors:
        return jsonify(error="No valid face data received. Please scan your face again."), 400

    photo = None
    photo_url = str(body.get("photo", ""))
    if photo_url.startswith("data:image/jpeg;base64,"):
        try:
            photo = base64.b64decode(photo_url.split(",", 1)[1], validate=True)
        except ValueError:
            photo = None
        if photo and len(photo) > MAX_PHOTO_BYTES:
            return jsonify(error="Photo is too large."), 400

    existing = _best_match(descriptors)
    if existing and existing[2] < MATCH_THRESHOLD:
        return jsonify(error=f"This face is already registered as “{existing[1]}”."), 409

    pid = uuid.uuid4().hex
    db.add_person(pid, name, descriptors, photo)
    return jsonify(ok=True, id=pid, name=name)


@app.post("/api/verify")
def verify():
    body = request.get_json(silent=True) or {}
    probes = _clean_descriptors(body.get("descriptors"), max_count=5)
    if not probes:
        return jsonify(error="No valid face data received."), 400

    best = _best_match(probes)
    if best and best[2] < MATCH_THRESHOLD:
        pid, name, score = best
        return jsonify(match=True, name=name, id=pid, distance=round(score, 4),
                       photo=f"/api/people/{pid}/photo")
    return jsonify(match=False, distance=round(best[2], 4) if best else None)


@app.get("/api/people/<pid>/photo")
def photo(pid):
    data = db.get_photo(pid)
    if not data:
        abort(404)
    return Response(data, mimetype="image/jpeg", headers={"Cache-Control": "private, max-age=300"})


@app.get("/api/people")
def people():
    _require_admin()
    return jsonify(people=db.list_people(), threshold=MATCH_THRESHOLD)


@app.delete("/api/people/<pid>")
def delete(pid):
    _require_admin()
    db.delete_person(pid)
    return jsonify(ok=True)


@app.get("/api/stats")
def stats():
    return jsonify(enrolled=len(db.all_descriptors()))


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=int(os.environ.get("PORT", "5000")), debug=True, use_reloader=False)
