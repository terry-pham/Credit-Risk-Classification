

def plotHistNumerical(data, columns, hue = None, bins = 50, col_wrap=5):
    import seaborn as sns
    from matplotlib import pyplot as plt
    # Plot histograms of numerical columns
    melted = None
    
    if hue is None:
        melted = data[columns].melt(var_name='cols',value_name='vals')
    else:
        melted = data[columns + [hue]].melt(hue,var_name='cols',value_name='vals')

    melted['vals'] = pd.to_numeric(melted['vals'].tolist())

    g = sns.FacetGrid(melted,col='cols',col_wrap=5,sharex=False,sharey=False,legend_out=True, hue = hue)
    g.map(plt.hist,"vals",bins=50)
    return g

def plotHistNumericalUnivariate(data,num_cols=None,hue=None,bins=50,col_wrap=5):
    # Plot histograms of numerical columns
    import seaborn as sns
    from matplotlib import pyplot as plt
    melted = None
    if hue is not None:
        melted = data[num_cols+[hue]].melt(hue,var_name='cols',value_name='vals')
    else:
        melted = data[num_cols].melt(var_name='cols',value_name='vals')

    melted['vals'] = pd.to_numeric(melted['vals'].tolist())

    g = sns.FacetGrid(melted,col='cols',col_wrap=col_wrap,sharex=False,sharey=False,legend_out=True,hue=hue)
    g.map(plt.hist,"vals",bins=bins)
    
    return g


def plotBoxPlot(data, columns, label = None, col_wrap = 5):
    import seaborn as sns
    from matplotlib import pyplot as plt
    # Plot histograms of numerical columns across labels

    # Plot histograms of numerical columns
    melted = None
    
    if label is not None:
        melted = data[columns+[label]].melt(label,var_name='cols',value_name='vals')
    else: 
        melted = data[columns].melt(var_name='cols',value_name='vals')

    g = sns.FacetGrid(melted,col='cols',col_wrap=5,sharex=False,sharey=False,legend_out=True)
    g.map(sns.boxplot,label,'vals').add_legend()
    
    return g


def plotPR_AUC(target,pred_proba, pos_label = 1):
    from sklearn import metrics
    from matplotlib import pyplot as plt

    # metrics.precision_recall_fscore_support(target,pred)

    precision, recall, _ = metrics.precision_recall_curve(target, pred_proba, pos_label=pos_label)
    roc_auc = metrics.auc(recall, precision)

    plt.figure()
    lw = 2
    plt.plot(recall,precision, color='darkorange',
             lw=lw, label='PR curve (area = %0.2f)'% roc_auc)
    # plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
    plt.xlim([-0.02, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title('ROC curve')
    plt.legend(loc="lower right")
    plt.show()

def plotROC_AUC(target,pred_proba, pos_label = 1, drop_intermediate = False):
    from sklearn import metrics
    from matplotlib import pyplot as plt

    metrics.roc_auc_score(target,pred_proba)

    fpr, tpr, _ = metrics.roc_curve(target, pred_proba, pos_label=pos_label, drop_intermediate = drop_intermediate)
    roc_auc = metrics.auc(fpr, tpr)
    #xgb.plot_importance(gbm)
    #plt.show()
    plt.figure()
    lw = 2
    plt.plot(fpr, tpr, color='darkorange',
             lw=lw, label='ROC curve (area = %0.2f)' % roc_auc)
    plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
    plt.xlim([-0.02, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC curve')
    plt.legend(loc="lower right")
    plt.show()