"""실사용 조건 분석 — 최종 점수가 아니라 사용자가 겪는 것을 잰다.

두 가지가 빠져 있었다.

① **통화 도중 경고.** 평가는 통화 최종 점수만 봤다. 그런데 사용자는
   통화 도중에 경고를 받는다. 3번째 마디에서 위험으로 떴다가 마지막에
   안전으로 끝나면 지금 평가는 통과로 세지만 사용자는 이미 경고음을 들었다.

② **실제 발생률.** 지금까지 사기:정상이 대략 1:1이었다. 실사용은 그 비율이
   아니다. 오탐률 1%도 발생률이 낮으면 정밀도를 무너뜨린다.
"""
import json, pathlib, sys
sys.path.insert(0, ".")
from mirinae.mode1.patterns import load_db
from mirinae.mode1.scorer import Scorer, CallState, THRESHOLD_ALERT, THRESHOLD_WARN

db = load_db(); sc = Scorer(db)
files = sorted(pathlib.Path("eval").glob("scenarios*.json"))
fraud, benign = [], []
for f in files:
    for s in json.loads(f.read_text(encoding="utf-8"))["scenarios"]:
        (fraud if s["label"] == "fraud" else benign).append((f.stem, s))

def trace(spec):
    st = CallState(sc)
    dv = float(spec.get("deepvoice_score", 0.0))
    peak_alert = peak_warn = False
    final = 0.0
    for u in spec["utterances"]:
        r = st.add_utterance(u["text"], deepvoice_score=dv)
        peak_alert |= r.score >= THRESHOLD_ALERT
        peak_warn |= r.score >= THRESHOLD_WARN
        final = r.score
    return final, peak_alert, peak_warn

print("=" * 72)
print("① 통화 도중 경고 — 최종 점수만 보면 놓치는 것")
print("=" * 72)
b_final_alert = b_peak_alert = b_peak_warn = 0
transient = []
for src, s in benign:
    final, pa, pw = trace(s)
    b_final_alert += final >= THRESHOLD_ALERT
    b_peak_alert += pa
    b_peak_warn += pw
    if pa and final < THRESHOLD_ALERT:
        transient.append((src, s["id"], s.get("title", "")))
n_b = len(benign)
print(f"  정상 통화 {n_b}건")
print(f"    최종 점수가 '위험'          {b_final_alert:3}건  ({b_final_alert/n_b*100:5.2f}%)")
print(f"    통화 **도중** '위험'을 지남  {b_peak_alert:3}건  ({b_peak_alert/n_b*100:5.2f}%)  ← 사용자가 겪는 값")
print(f"    통화 도중 '주의' 이상        {b_peak_warn:3}건  ({b_peak_warn/n_b*100:5.2f}%)")
if transient:
    print("\n  최종은 안전인데 도중에 경고가 떴던 통화:")
    for src, i, t in transient: print(f"    {i:10} {t[:44]}  ({src})")
else:
    print("\n  최종은 안전인데 도중에 경고가 뜬 통화: 없음")

f_alert = sum(trace(s)[1] for _, s in fraud)
n_f = len(fraud)
tpr = f_alert / n_f
fpr = b_peak_alert / n_b
print()
print("=" * 72)
print("② 실제 발생률 기준 정밀도")
print("=" * 72)
print(f"  탐지율(도중 포함) {tpr*100:.1f}%  ({f_alert}/{n_f})   오탐률(도중 포함) {fpr*100:.2f}%")
print()
print("  통화 1000건 중 사기가 N건일 때 — 경고가 울리면 진짜일 확률")
print("  " + "-" * 62)
print(f"  {'사기 비율':>10} {'경고 횟수':>10} {'진짜':>7} {'헛경고':>7} {'정밀도':>9}")
for per_1000 in (500, 100, 20, 5, 1):
    p = per_1000 / 1000
    tp = 1000 * p * tpr
    fp = 1000 * (1 - p) * fpr
    prec = tp / (tp + fp) if tp + fp else 0.0
    print(f"  {per_1000:>5}/1000 {tp+fp:>10.1f} {tp:>7.1f} {fp:>7.1f} {prec*100:>8.1f}%")
print()
print("  ※ 사기 비율 500/1000은 지금까지의 평가 세트 구성이다. 실사용은 아래쪽이다.")
