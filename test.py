# -*- coding: utf-8 -*-
"""
2026년 테스트 실행 파일

실행: py test.py
함께 둘 파일: final_original.py, common_data_13cols.xlsx, previous_config.json

읽는 순서
1. 개발자료(2016~2023)와 새 테스트(2026)를 구분합니다.
2. 개발자료 안에서 시간순 검증으로 임계값을 재현합니다.
3. 개발자료 전체로 모델을 학습하고 설정을 고정합니다.
4. 2026년 정답을 학습에 넣지 않고 예측과 정답을 비교합니다.
5. 날짜별 결과, 계절별 결과, 점수 구간과 SHAP를 저장합니다.

주의: 저장 모델만 읽는 방식이 아니라 개발 단계도 재현하는 평가 코드입니다.
2026년 성능에 맞춰 변수나 임계값을 다시 조정하지 않습니다.
"""
from pathlib import Path
import sys,json,hashlib,zipfile,datetime,html
import importlib.util
OUT=Path(__file__).resolve().parent
# Install the packages listed in requirements.txt before running.
import numpy as np,pandas as pd
from sklearn.metrics import average_precision_score,roc_auc_score,f1_score,brier_score_loss
spec=importlib.util.spec_from_file_location('frozen_ci',OUT/'final_original.py');m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)
# [1] 원본 코드의 자료 읽기 함수를 사용합니다. dev에는 2016~2023년만 들어갑니다.
full,dev=m.load_data(str(OUT/'common_data_13cols.xlsx'))
assert len(dev)==445 and dev.year.max()==2023
assert (full.year==2026).sum()==44
# Reproduce design on development data only; never pass 2026 to selection.
selection=[];oofs={}
# 13열 최종본에서는 TOC_last만 남겨 두었습니다.
# 없는 과거 후보 열을 요구하지 않도록, 실제 존재하는 TOC 열만 읽습니다.
for toc in [c for c in m.TOC_CANDIDATES if c in dev.columns]:
 oofs[toc]=m.rolling_oof(dev,m.BASE_FEATURES+[toc]);selection.append(m.summarize(toc,oofs[toc]))
sel=pd.DataFrame(selection);default=m.TOC_CANDIDATES[0]
best=float(sel.AP_mean.max());da=float(sel.loc[sel.model.eq(default),'AP_mean'].iloc[0])
toc=default if da>=best-m.TOC_RULE_TOLERANCE else sel.sort_values('AP_mean',ascending=False).model.iloc[0]
features=m.BASE_FEATURES+[toc];oof=oofs[toc]
# [2] 검증 예측에서 위험 탐지율 85% 이상을 만족하면서 정상 오경보율이 가장 낮은 기준을 찾습니다.
# 여기에는 2026년 자료가 들어가지 않습니다.
threshold=m.select_threshold(oof[m.TARGET].to_numpy(),oof.score.to_numpy(),m.TARGET_RECALL)
prior=json.loads((OUT/'previous_config.json').read_text(encoding='utf-8'))
print('Development reproduction:',features,threshold,'previous threshold:',prior['threshold'],flush=True)
assert features==prior['features']
# [3] 확정 하이퍼파라미터로 개발자료 445건을 학습합니다.
model=m.build_model(dev[m.TARGET].to_numpy()).fit(dev[features],dev[m.TARGET])
model.save_model(OUT/'frozen_model.json')
config={'features':features,'threshold':threshold,'params':m.XGB_PARAMS,'training_period':'2016-2023','training_n':len(dev),'test_period':'2026-01-01 to 2026-09-30','selection':'2020-2023 rolling OOF; reproduced before 2026 predictions','n_boot':500,'block_size':6,'score_note':'uncalibrated model score, not event probability','source_code_sha256':hashlib.sha256((OUT/'final_original.py').read_bytes()).hexdigest(),'data_sha256':hashlib.sha256((OUT/'common_data_13cols.xlsx').read_bytes()).hexdigest(),'model_sha256':hashlib.sha256((OUT/'frozen_model.json').read_bytes()).hexdigest(),'frozen_at':datetime.datetime.now().isoformat()}
config['previous_run_threshold']=prior['threshold'];config['threshold_reproduction_note']='Current supplied code on 2016-2023 yields 0.335985; previous run yielded 0.351872. Current result frozen before test. Historical inputs and labels match.'
(OUT/'frozen_config.json').write_text(json.dumps(config,ensure_ascii=False,indent=2),encoding='utf-8')
sel.to_csv(OUT/'development_selection_reproduction.csv',index=False,encoding='utf-8-sig')
print('Design frozen',toc,threshold,flush=True)
# [4] 학습이 끝난 다음 2026년 44건을 꺼냅니다.
# 결측 입력 4건도 기존 XGBoost 결측 분기 방식으로 처리합니다.
test=full.loc[full.year.eq(2026)].copy().sort_values('target_date').reset_index(drop=True)
# 엑셀에서 관리용 sample_id를 제외했으므로 날짜로 같은 식별자를 복원합니다.
# 모델 입력변수가 늘어나는 것은 아닙니다.
test['sample_id']=test['target_date'].dt.strftime('CS_%Y%m%d')
test['score']=model.predict_proba(test[features])[:,1];test['alert']=(test.score>=threshold).astype(int)
test['missing_inputs']=test[features].isna().apply(lambda r:', '.join(r.index[r]),axis=1)
test['complete_inputs']=test.missing_inputs.eq('')
test['season']=test.target_date.dt.month.map({1:'겨울(1~2월)',2:'겨울(1~2월)',3:'봄(3~5월)',4:'봄(3~5월)',5:'봄(3~5월)',6:'여름(6~8월)',7:'여름(6~8월)',8:'여름(6~8월)',9:'가을(9월만)'})
# [5] 실제 위험·정상과 예측 경보를 비교해 TP/FP/FN/TN 및 성능을 계산합니다.
# Recall은 실제 위험 중 찾아낸 비율, Precision은 경보 중 실제 위험 비율입니다.
def summarize_test(label,d):
 y=d[m.TARGET].to_numpy();s=d.score.to_numpy();r=m.metrics_row(y,s,threshold)
 r.update(group=label,n=len(d),positive=int(y.sum()),negative=int((y==0).sum()),F1=float(f1_score(y,s>=threshold,zero_division=0)),AP=float(average_precision_score(y,s)) if y.sum()>0 else None,ROC_AUC=float(roc_auc_score(y,s)) if len(np.unique(y))==2 else None,Brier=float(brier_score_loss(y,s)))
 return r
rows=[summarize_test('2026 전체 44건',test),summarize_test('입력 완비 40건(보조)',test[test.complete_inputs])]
for season in ['겨울(1~2월)','봄(3~5월)','여름(6~8월)','가을(9월만)']:
 d=test[test.season.eq(season)]
 if len(d):rows.append(summarize_test(season,d))
metrics=pd.DataFrame(rows);metrics.to_csv(OUT/'test_metrics.csv',index=False,encoding='utf-8-sig')
monthly=pd.DataFrame([summarize_test(f'{month}월',d) for month,d in test.groupby(test.target_date.dt.month)])
monthly.to_csv(OUT/'monthly_metrics.csv',index=False,encoding='utf-8-sig')
print(metrics[['group','n','TP','FP','FN','TN','Recall','Precision','F1','AP']].to_string(index=False),flush=True)
print('Computing 500 training-only block bootstrap fits for score intervals...',flush=True)
# [6] 개발자료의 연속된 관측행을 묶어서 500번 다시 학습해 점수의 흔들림을 추정합니다.
# 2026년은 예측 대상으로만 사용하며, 이 구간은 발생확률의 신뢰구간이 아닙니다.
lo,hi,sd=m.fit_mbb_predict(dev[features],dev[m.TARGET].to_numpy(),test[features],n_boot=500,block_size=6,seed=m.SEED)
test['score_ci_lo']=lo;test['score_ci_hi']=hi;test['score_boot_sd']=sd
test['score_zone']=np.where(hi<threshold,'점수 낮음',np.where(lo>threshold,'점수 높음','판정 경계'))
test['outcome']=np.select([(test.alert==1)&(test[m.TARGET]==1),(test.alert==1)&(test[m.TARGET]==0),(test.alert==0)&(test[m.TARGET]==1)],['TP','FP','FN'],default='TN')
# [7] SHAP로 각 입력변수가 점수에 얼마나 기여했는지 기록합니다. 인과관계 증명은 아닙니다.
contrib=model.get_booster().predict(m.xgb.DMatrix(test[features],feature_names=features),pred_contribs=True)
for i,f in enumerate(features):test['SHAP_'+f]=contrib[:,i]
assert np.allclose(contrib.sum(axis=1),model.get_booster().predict(m.xgb.DMatrix(test[features],feature_names=features),output_margin=True),atol=1e-5)
# [8] 날짜별 예측·정답·결측·점수 구간·SHAP를 CSV로 저장합니다.
predcols=['sample_id','target_date','prediction_at','cutoff_date',m.TARGET,'Cyano_target_cells_mL','score','alert','outcome','score_ci_lo','score_ci_hi','score_zone','complete_inputs','missing_inputs','season']+features+['SHAP_'+f for f in features]
test[predcols].to_csv(OUT/'test_predictions.csv',index=False,encoding='utf-8-sig')
pd.DataFrame({'feature':features,'mean_abs_SHAP_2026':np.abs(contrib[:,:-1]).mean(axis=0)}).sort_values('mean_abs_SHAP_2026',ascending=False).to_csv(OUT/'test_SHAP_importance.csv',index=False,encoding='utf-8-sig')
primary=rows[0];note=['2026년 성능으로 변수·임계값·하이퍼파라미터를 조정하지 않았다.','44건 전체가 주 평가이며, 결측 4건은 XGBoost 고유 결측 분기로 예측했다. 40건은 보조 평가다.','원점수는 보정된 발생확률이 아니다. 부트스트랩 구간은 학습 표본 변동에 따른 점수 불확실성이다.','신뢰구간의 Wilson 계산은 이항 독립 근사다. 시계열 의존성과 임계값 선정 불확실성을 모두 반영하지 않는다.','1~9월 중 남조류 수치가 공개된 날짜만 포함되며, 10~12월 및 사계절 전체 검증이 아니다.','수질 실제 공개시각 미확인·유량 잠정값·2025년 이후 칠서 채수지점 변경을 고려해야 한다. 후향적 새 연도 평가이며 실시간 운영 실증이 아니다.','2024~2025년은 과거 모델 개발 과정에서 열람했으므로 독립 테스트라는 표현을 사용하지 않는다.','SHAP는 예측 기여이며 TOC·가뭄의 인과효과를 입증하지 않는다.']
(OUT/'results.json').write_text(json.dumps({'primary':primary,'all_groups':rows,'limitations':note},ensure_ascii=False,indent=2,allow_nan=True),encoding='utf-8')
note.append('이번 첨부 코드의 개발자료 재실행 임계값은 0.335985이며 이전 저장 실행본의 0.351872와 다르다. 기존 입력값·정답의 일치는 확인했다. 이번 값은 2026년 예측 전에 고정했으며 테스트 결과에 맞춰 선택한 값이 아니다.')
# Standalone visual report, without requiring extra plotting dependencies.
tp,fp,fn,tn=[primary[k] for k in ['TP','FP','FN','TN']]
svg=f'''<svg xmlns="http://www.w3.org/2000/svg" width="680" height="330" viewBox="0 0 680 330"><style>text{{font-family:Malgun Gothic,Arial;fill:#172b3a;font-size:16px}}.big{{font-size:38px;font-weight:bold}}</style><rect width="680" height="330" fill="white"/><text x="24" y="30">2026년 테스트 · 실제와 예측 비교 (44건)</text><text x="220" y="65">예측 정상</text><text x="430" y="65">예측 위험</text><text x="24" y="135">실제 정상</text><text x="24" y="245">실제 위험</text><rect x="180" y="80" width="200" height="100" fill="#e5eff7"/><rect x="390" y="80" width="200" height="100" fill="#fff0dc"/><rect x="180" y="190" width="200" height="100" fill="#fde5e5"/><rect x="390" y="190" width="200" height="100" fill="#dcefe9"/><text class="big" x="240" y="135">{tn}</text><text x="245" y="163">TN</text><text class="big" x="450" y="135">{fp}</text><text x="430" y="163">오경보 FP</text><text class="big" x="240" y="245">{fn}</text><text x="220" y="273">미탐지 FN</text><text class="big" x="450" y="245">{tp}</text><text x="435" y="273">탐지 TP</text></svg>'''
(OUT/'confusion_matrix.svg').write_text(svg,encoding='utf-8')
display=['group','n','positive','negative','TP','FP','FN','TN','Recall','Precision','F1','AP','ROC_AUC','FPR']
report=f'''<!doctype html><html lang="ko"><meta charset="utf-8"><title>2026년 새 연도 테스트 결과</title><style>body{{font:16px/1.85 'Malgun Gothic',sans-serif;color:#182b39;max-width:1180px;margin:48px auto;padding:0 28px}}h1{{font-size:29px}}h2{{margin-top:38px;border-bottom:2px solid #243746}}table{{border-collapse:collapse;width:100%;font-size:14px}}td,th{{border:1px solid #bbb;padding:9px;text-align:right}}th{{background:#eef3f6}}td:first-child{{text-align:left}}.note{{background:#f1f5f7;padding:18px}}img{{max-width:100%}}</style><h1>2026년 1~9월 새 연도 테스트 결과</h1><p>학습: 2016~2023년 445건 · 테스트: 2026년 실측 정답 44건 · 임계값: {threshold:.9f}</p><h2>1. 고정한 모델과 평가 원칙</h2><p>입력: {', '.join(features)}. 첨부 확정 코드의 개발자료 선택 절차를 재현하여 확정 7변수와 개발자료에서 산출한 임계값을 2026년 예측 전에 고정했다. 2026년은 재학습·임계값 선정에 사용하지 않았다.</p><h2>2. 전체 및 계절별 결과</h2>{metrics[display].to_html(index=False,float_format=lambda x:f'{x:.3f}',na_rep='평가 불가',border=0)}<p>Recall: 실제 위험 중 탐지한 비율. Precision: 경보 중 실제 위험 비율. FPR: 실제 정상 중 오경보 비율. AP: 여러 임계값에서 위험 순위 구분을 종합한 지표.</p><img src="confusion_matrix.svg"><h2>3. 날짜별 예측</h2><p>원점수와 구간은 발생확률이 아니다. 구간이 임계값과 겹치는 날짜를 판정 경계로 표시했다.</p>{test[['target_date',m.TARGET,'score','alert','outcome','score_ci_lo','score_ci_hi','score_zone','missing_inputs']].to_html(index=False,float_format=lambda x:f'{x:.3f}',border=0)}<h2>4. 해석과 한계</h2><ul>{''.join('<li>'+html.escape(x)+'</li>' for x in note)}</ul><p>원본 코드와 고정 설정, 날짜별 결과 CSV를 함께 제공한다. 원자료의 관측일 정렬과 SPI30 산출은 공통데이터 추가 시 적용한 기준을 따른다.</p></html>'''
(OUT/'2026_test_report.html').write_text(report,encoding='utf-8')
files=['test_metrics.csv','monthly_metrics.csv','test_predictions.csv','test_SHAP_importance.csv','results.json','frozen_config.json','frozen_model.json','final_original.py','test.py','confusion_matrix.svg','2026_test_report.html','development_selection_reproduction.csv']
with zipfile.ZipFile(OUT/'2026_test_results.zip','w',zipfile.ZIP_DEFLATED) as z:
 for f in files:z.write(OUT/f,f)
print('DONE',flush=True)
