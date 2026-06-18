from sklearn.model_selection import RandomizedSearchCV,GridSearchCV
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.metrics import precision_score, recall_score, f1_score
from sklearn.metrics import RocCurveDisplay,roc_curve,auc
import seaborn as sns
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt

def predict(X_test,model):
    return model.predict(X_test)


def roc_curve_display(X_test, y_test, gs_log_reg, gs_random_forest):
    #Plot ROC curve and calculate AUC
    y_scores=gs_log_reg.predict_proba(X_test)[:,1]
    fpr, tpr,_=roc_curve(y_test,y_scores)
    roc_auc=auc(fpr,tpr)

    y2_scores=gs_random_forest.predict_proba(X_test)[:,1]
    fpr2,tpr2,_=roc_curve(y_test, y2_scores)
    roc_auc2=auc(fpr2,tpr2)

    plt.figure()
    plt.plot(fpr,tpr,label=f'Logistic Regression (AUC={roc_auc:.2f})')
    plt.plot(fpr2,tpr2,label=f'Random Forest (AUC={roc_auc2:.2f})')
    plt.plot([0, 1], [0, 1], 'r--', label='Random Guess')
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("ROC Curve and AUC for Logistic Regression/Random Forest")
    plt.tight_layout()
    plt.legend()
    plt.show()


def confusion_mat(y_test,y_preds,y_preds2):
    fig, ax =plt.subplots(1,2,figsize=(3,3))
    sns.heatmap(confusion_matrix(y_test, y_preds),
                annot=True,
                ax=ax[0])
    sns.heatmap(confusion_matrix(y_test, y_preds2),
                annot=True,
                ax=ax[1])
    fig.set_size_inches(12, 5)
    ax[0].set_xlabel("True label")
    ax[0].set_ylabel("Predicted label")

    ax[1].set_xlabel("True label")
    ax[1].set_ylabel("Predicted label")

    ax[0].set_title("Logistic Regression Heatmap")
    ax[1].set_title("Random Forest  Heatmap")
    plt.show();