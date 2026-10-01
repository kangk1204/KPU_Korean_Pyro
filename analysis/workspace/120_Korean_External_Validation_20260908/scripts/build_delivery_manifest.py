"""Hash the delivered fallback package after all reports have been finalized."""
from pathlib import Path
import hashlib,json

def require(condition, message="Integrity check failed"):
    if not condition:
        raise RuntimeError(message)

ROOT=Path(__file__).resolve().parents[1]
def digest(path):
 h=hashlib.sha256()
 with path.open('rb') as f:
  for chunk in iter(lambda:f.read(1024*1024),b''):h.update(chunk)
 return h.hexdigest()
def files(base):
 return sorted(p for p in base.rglob('*') if p.is_file() and not any(x.startswith('.') or x=='__pycache__' for x in p.relative_to(base).parts) and not p.name.startswith('~$'))
def write(base):
 target=base/'SHA256SUMS.txt'
 rows=[(digest(p),str(p.relative_to(base))) for p in files(base) if p!=target]
 target.write_text(''.join(f'{h}  {name}\n' for h,name in rows))
 require(all((digest(base / name) == h for (h, name) in rows)), 'Integrity check failed: all((digest(base / name) == h for (h, name) in rows))')
 return len(rows)
if __name__=='__main__':
 print(json.dumps({'submission_files_verified':write(ROOT/'submission'),'package_files_verified':write(ROOT)},indent=2))
