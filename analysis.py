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
    precision=y[order[:n]].mean(); baseline=y.mean()
    metrics={"train_rows":len(train),"test_rows":len(test),"test_conversion_rate":baseline,
      "roc_auc":roc_auc_score(y,scores),"average_precision":average_precision_score(y,scores),
      "top_20_percent_precision":precision,"top_20_percent_lift":precision/baseline,"features":FEATURES}
    fig,axes=plt.subplots(1,2,figsize=(11,4.5))
    x=np.arange(1,len(y)+1)/len(y)
    axes[0].plot(x,np.cumsum(y[order])/y.sum(),label="Model ranking"); axes[0].plot([0,1],[0,1],"--",label="Random expectation")
    axes[0].set(title="Cumulative response capture",xlabel="Share of contacts",ylabel="Share of positive outcomes"); axes[0].legend()
    axes[1].bar(["All test contacts","Top 20% scored"],[baseline*100,precision*100])
    axes[1].set(title="Observed response concentration",ylabel="Positive response (%)")
    finish(args.output_dir,"Campaign Targeting",metrics,
      [f"Top-20% precision is {precision:.1%}, versus {baseline:.1%} in the held-out population ({precision/baseline:.2f}x lift).",
       f"Holdout ROC AUC: {metrics['roc_auc']:.3f}; average precision: {metrics['average_precision']:.3f}.",
       "Treat the ranking as a pilot candidate, subject to consent, fairness review and an incremental-effect experiment."],
      ["First 80% of original ordered rows train, last 20% test; the dataset has no complete timestamp for every record.",
       "Duration, current-campaign contact count, demographics and current contact timing are excluded. Remaining financial variables may still create fairness risks.",
       "No profit or causal uplift claim; historic selection and temporal shifts affect results."],fig)
if __name__=="__main__": main()

