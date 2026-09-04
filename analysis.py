from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, average_precision_score
from utils import arguments, dataset, theme, finish, plt, np
FEATURES=["balance","default","housing","loan","previous","pdays","poutcome"]
def main():
    args=arguments(); theme(); d=dataset("bank",args.data_dir)
    assert set(d.y.unique())=={"yes","no"}
    split=int(len(d)*.8); train,test=d.iloc[:split],d.iloc[split:]
    categorical=["default","housing","loan","poutcome"]; numeric=["balance","previous","pdays"]
    prep=ColumnTransformer([("categorical",OneHotEncoder(handle_unknown="ignore"),categorical),("numeric",StandardScaler(),numeric)])
    model=make_pipeline(prep,LogisticRegression(max_iter=2000,random_state=42))
    model.fit(train[FEATURES],train.y.eq("yes").astype(int))
    y=test.y.eq("yes").astype(int).to_numpy(); scores=model.predict_proba(test[FEATURES])[:,1]
    assert "duration" not in FEATURES and "campaign" not in FEATURES
    order=np.argsort(-scores,kind="stable"); n=max(1,int(len(y)*.2))
    ranked=y[order]; cumulative=np.cumsum(ranked); x=np.arange(1,len(y)+1)/len(y)
    precision_curve=cumulative/np.arange(1,len(y)+1); baseline=y.mean(); lift_curve=precision_curve/baseline
    precision=precision_curve[n-1]; capture=cumulative[n-1]/y.sum()
    curve=[]
    for decile in range(1,11):
        idx=max(1,len(y)*decile//10)-1
        curve.append({"target_contact_share":decile/10,"contacts":idx+1,"contact_share":x[idx],"precision":precision_curve[idx],"lift":lift_curve[idx],"response_capture":cumulative[idx]/y.sum()})
    assert np.isclose(curve[1]["precision"],precision)
    metrics={"train_rows":len(train),"test_rows":len(test),"test_conversion_rate":baseline,
      "roc_auc":roc_auc_score(y,scores),"average_precision":average_precision_score(y,scores),
      "top_20_percent_precision":precision,"top_20_percent_lift":precision/baseline,
      "top_20_percent_response_capture":capture,"contact_share_checkpoints":curve,"features":FEATURES}
    fig,axes=plt.subplots(1,2,figsize=(11,4.5))
    gains=cumulative/y.sum()
    axes[0].plot(x,gains,label="Model ranking"); axes[0].plot([0,1],[0,1],"--",color="#94a3b8",label="Random expectation")
    axes[0].axvline(x[n-1],color="#192f4b",linestyle=":")
    axes[0].scatter(x[n-1],capture,color="#192f4b",zorder=3)
    axes[0].annotate(f"20% contacts capture {capture:.1%}",xy=(x[n-1],capture),xytext=(.34,.48),arrowprops={"arrowstyle":"->","color":"#192f4b"},fontsize=8)
    axes[0].set(title="Cumulative response capture",xlabel="Share of contacts",ylabel="Share of positive outcomes"); axes[0].legend()
    visible=x>=.01
    axes[1].plot(x[visible],lift_curve[visible],color="#087f8c",linewidth=2)
    axes[1].axhline(1,color="#94a3b8",linestyle="--",label="Population average")
    axes[1].axvline(x[n-1],color="#192f4b",linestyle=":")
    axes[1].scatter(x[n-1],lift_curve[n-1],color="#192f4b",zorder=3)
    axes[1].annotate(f"20%: {lift_curve[n-1]:.2f}× lift\nprecision {precision:.1%}",xy=(x[n-1],lift_curve[n-1]),xytext=(.39,lift_curve[n-1]+.35),arrowprops={"arrowstyle":"->","color":"#192f4b"},fontsize=8)
    axes[1].set(title="Lift as contact budget expands",xlabel="Share of contacts",ylabel="Lift versus test population",xlim=(0,1))
    axes[1].legend(fontsize=8)
    finish(args.output_dir,"Campaign Targeting",metrics,
      [f"Top-20% precision is {precision:.1%}, versus {baseline:.1%} in the held-out population ({precision/baseline:.2f}x lift).",
       f"Holdout ROC AUC: {metrics['roc_auc']:.3f}; average precision: {metrics['average_precision']:.3f}.",
       "Treat the ranking as a pilot candidate, subject to consent, fairness review and an incremental-effect experiment."],
      ["First 80% of original ordered rows train, last 20% test; the dataset has no complete timestamp for every record.",
       "Duration, current-campaign contact count, demographics and current contact timing are excluded. Remaining financial variables may still create fairness risks.",
       "No profit or causal uplift claim; historic selection and temporal shifts affect results."],fig)
if __name__=="__main__": main()
