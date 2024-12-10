import gzip
import math
import statistics
import pandas
import numpy


# read data from a text file
data = pandas.read_table("data/wheat/data.txt")
# fix the tabulation
text = open("data/wheat/data.txt").read()
text = text.replace("\t\t", "\t")
# in-memory file-like object
import io
data = pandas.read_table(io.StringIO(text))
# add column names
names = "area perimeter compactness length width asymmetry groove variety".split()
data = pandas.read_table(io.StringIO(text), names=names)

# write data frame to a CSV file
data.to_csv("processed/wheat.csv")

data.columns
data.index
data.values

# selecting columns
data.age
data["age"]

# selecting rows
data.iloc[3]
data[0:3]

# slicing rows and columns
data.iloc[3:5, 0:2]
data.loc[[1, 2, 5], ["age", "sex"]]

for name in "anaemia diabetes hypertension sex smoking died".split():
    data[name] = data[name].astype("category")
# ordinal
pandas.Categorical(data.ejection, ordered=True)

# binning into groups
pandas.cut(data.age, range(0, 105, 20))

# quick look
data.head()
data.tail()

# dimensions
data.shape

# summaries
data.info()
data.describe()
# show frequency
data.sex.value_counts()
data.ejection.value_counts()

# show patients with short follow-up
data.query("time < 10")

# complex query
data.query("time > 100 and died == 1")

# sorting
data.sort_values(by="time")

data.CPK / 10

# transformation with function
data.ejection.apply(math.sqrt)
data.ejection.apply(lambda x: "yes" if x > 40 else "no")

# min-max normalisation to [0, 1] range
(data.CPK - data.CPK.min()) / (data.CPK.max() - data.CPK.min())
data.to_pickle("processed/heart.pkl.gz")
pandas.read_csv("data/mushroom/data.csv.gz")
# read file line by line to a list
lines = open("data/mushroom/description.txt").readlines()

# names = [line.split()[1] in lines if line.startswith("-")]
names = [name[:-1].replace("-", "_") for name in names]

pandas.read_csv("data/mushroom/data.csv.gz", names=names)
names.insert(0, "kind")
data = pandas.read_csv("data/mushroom/data.csv.gz", names=names, na_values="?")
for column in data:
    data[column] = data[column].astype("category")

data.to_pickle("processed/mushroom.pkl.gz")

data = pandas.read_csv("data/psoriasis/data.csv")

# missing per attribute
data.BASELINE_CRP.isna()
data.BASELINE_CRP.isna().sum()

data.BASELINE_CRP.unique()
# total missing
data.isna()

data.isna().sum(axis=1)
(data.isna().sum(axis=1) > 0).sum()

data.isna().sum(axis=0)
# count unique values in each column
sorted(data[x].nunique() for x in data)

# categorical?
{x for x in data if data[x].nunique() < 10}

# rows
data.dropna(axis=0)

# columns
data.dropna(axis=1)
# fill with a single "dummy" value
data.fillna(99)

# fill with some distribution statistic
data.BASELINE_CRP.fillna(data.BASELINE_CRP.mean())
data.BASELINE_CRP.fillna(data.BASELINE_CRP.mode()[0])

data.BASELINE_CRP.value_counts()

data = pandas.read_excel("data/psoriasis/scores.xlsx")

data.isna().sum(axis=0)

data.dropna(how="all")
# forward fill
data.fillna(method="ffill")

# transpose data frame
data.T
# interpolation
data.interpolate(...)

data.interpolate(limit_direction="?", limit_area="?")

data = pandas.read_pickle("processed/heart.pkl.gz")

# statistics
data.ejection.describe()
data.ejection.median()
data.ejection.quantile(0.75)
# simple aggregation
data.ejection.sum()
data.ejection.value_counts()
# grouping
g = data.groupby("age")
g.groups
g.get_group(70)

# per group aggregation
g.time.median()
g.ejection.aggregate([statistics.mean, statistics.mode])

# transformation
data.groupby("sex").CPK.describe()

plt.plot(x, y)
plt.xlabel("x")
plt.yscale("log")
plt.title("Figure 1")
plt.legend()

plt.show()

fig, p = plt.subplots()

x = numpy.linspace(0, 10, 100)
y = numpy.sin(x)

p.plot(x, y)
p.set_xlabel("x")
p.set_yscale("log")
p.set_title("Figure 1")

p.legend()
p.grid(True)

fig.savefig("results/figure.png", dpi=150, bbox_inches="tight")

# bar
p.barh(["no", "yes"], data.groupby("anaemia").CPK.median())

# scatter
# can use data frame and vary colour and size
p.scatter("ejection", "serum_sodium", c="died", s="time", data=data)
# histogram
p.hist(data.age, 60, density=True, facecolor="g", alpha=0.75)
# subplots
fig, (p1, p2) = plt.subplots(1, 2)
p1.barh(["no", "yes"], data.groupby("diabetes").size(), color="g")
p2.hist(data.ejection, 10, color="k")

import seaborn

seaborn.set_style("whitegrid")
seaborn.set_context("paper")
seaborn.set_palette("deep", color_codes=True)
# histogram
grid = seaborn.displot(x="age", data=data)
grid.savefig("results/figure.png", dpi=150, bbox_inches="tight")

grid = seaborn.displot(x="sex", data=data)
# kernel density estimation
grid = seaborn.displot(x="age", kde=True, data=data)

# independent KDE plot
grid = seaborn.displot(x="time", kind="kde", fill=True, data=data)
# more variants
grid = seaborn.displot(x="age", hue="sex", data=data)
grid = seaborn.displot(x="age", hue="sex", multiple="stack", data=data)
grid = seaborn.displot(x="age", hue="sex", multiple="dodge", data=data)

data = pandas.read_csv("processed/wheat.csv")
grid = seaborn.catplot(x="variety", y="area", data=data)
grid = seaborn.catplot(x="variety", y="area", kind="swarm", data=data)
grid = seaborn.catplot(x="variety", y="compactness", kind="box", data=data)
grid = seaborn.catplot(x="variety", y="compactness", kind="violin", data=data)
# direct data pass, wide-form
grid = seaborn.catplot(data=data, kind="box", orient="h")
# confidence interval
from scipy import stats

sample = data.query("variety == 1").width
mu, sigma, n = sample.mean(), sample.std(), len(sample)
stats.norm.interval(0.95, loc=mu, scale=sigma / math.sqrt(n))

# stats.t.interval(0.95, loc=mu, scale=sigma / math.sqrt(n), df=len(sample) - 1)

seaborn.scatterplot(x="perimeter", y="length", data=data, ax=p)

seaborn.scatterplot(x="perimeter", y="length", hue="variety", data=data, ax=p)
# regression plot with confidence interval estimated with bootstrap
seaborn.regplot(x="compactness", y="length", data=data, ax=p)
data.perimeter.corr(data.length)
seaborn.heatmap(data.corr(), annot=True, fmt=".2f", ax=p)
for label in p.get_xticklabels():
    label.set_rotation(30)


data = pandas.read_csv("processed/wheat.csv")

grid = seaborn.JointGrid(x="width", y="length", data=data)
grid.plot_joint(seaborn.scatterplot, alpha=0.5, edgecolor="b")
grid.plot_marginals(seaborn.kdeplot, fill=True)

grid.savefig("results/figure.png", dpi=150, bbox_inches="tight")
grid = seaborn.JointGrid(x="width", y="length", hue="variety", data=data)
grid = seaborn.JointGrid(x="width", y="length", data=data)
grid.plot(seaborn.regplot, seaborn.boxplot)
data = pandas.read_pickle("processed/mushroom.pkl.gz")

grid = seaborn.jointplot(x="habitat", y="cap_color", kind="hist", data=data)
grid = seaborn.jointplot(x="habitat", y="cap_color", hue="kind", kind="hist", data=data)

data = pandas.read_csv("processed/wheat.csv")

columns = ["area", "perimeter", "compactness", "asymmetry"]
grid = seaborn.PairGrid(vars=columns, data=data)
grid.map_diag(seaborn.histplot)
grid.map_offdiag(seaborn.scatterplot)
columns = ["width", "length", "compactness", "asymmetry"]
grid = seaborn.PairGrid(vars=columns, hue="variety", data=data)
grid.map_diag(seaborn.histplot, multiple="stack", element="step")
grid.map_offdiag(seaborn.scatterplot, size=data.area)
grid.add_legend(title="", adjust_subtitles=True)
grid = seaborn.pairplot(vars=columns, kind="scatter", diag_kind="hist", data=data)

data = pandas.read_pickle("processed/heart.pkl.gz")

grid = seaborn.displot(x="serum_sodium", hue="sex", multiple="dodge", data=data)
grid = seaborn.displot(x="serum_sodium", hue="sex", col="smoking",
    multiple="dodge", data=data)
grid = seaborn.displot(x="serum_sodium", hue="sex", col="smoking", row="hypertension",
    multiple="dodge", data=data)
grid = seaborn.FacetGrid(data, col="diabetes", row="hypertension")
grid.map(seaborn.scatterplot, "serum_sodium", "ejection", size=data.age)
grid.add_legend(title="age")
grid = seaborn.FacetGrid(data, col="diabetes", row="hypertension", hue="died")

scop = pandas.read_table("data/scop/scop_classification.txt.gz", sep=" ", skiprows=5)
columns = scop.columns
scop.dropna(axis=1, inplace=True)
# fix column names (shift left by 1)
columns_mapper = dict(zip(columns[:-1], columns[1:]))
scop.rename(columns_mapper, axis=1, inplace=True)
# select only the family information
scop["FA"] = scop.SCOPCLA.apply(lambda x: int(x[-7:]))
scop.loc[:, "FA FA-DOMID FA-PDBID FA-PDBREG".split()]
# select 10 largest families
top = scop.FA.value_counts()[:10].index
selection = scop.FA.isin(top)
scop = scop[selection]
# use family ID as index
scop.set_index("FA", inplace=True)

input_file = gzip.open("data/scop/scop_description.txt.gz", "rt")

ids = []
names = []

# read family ID and name
for line in input_file:
    if line.startswith("400"):
        ids.append(int(line[:7]))
        names.append(line[8:-1])
# use the lists to make a data frame
scop_names = pandas.DataFrame(data=names, index=ids, columns=["family_name"])
# scop.join?

scop = scop.join(scop_names)

scop["chain"] = scop["FA-PDBREG"].str[0]
scop["begin"] = scop["FA-PDBREG"].apply(lambda x: x.split(":")[1].split("-")[0])
scop["end"] = scop["FA-PDBREG"].apply(lambda x: x.split(":")[1].split("-")[1])

scop.drop(["FA-PDBREG", "FA-DOMID"], axis=1, inplace=True)
# remove rows with missing chain start position
scop = scop.loc[scop.begin.str.len() > 0, :]
scop.begin = scop.begin.astype(int)

# take only first part (before comma) of the multichain description
scop.end = scop.end.apply(lambda x: int(x.split(",")[0] if "," in x else x))
# combine PDB ID and chain as new index
scop = scop.set_index("FA-PDBID")
scop.index = scop.index + "_" + scop.chain.str.upper()

# take first chain fragment if there is more than one
scop = scop.groupby(level=0).first()

scop["sequence"] = ""
pdbid = None

from Bio import SeqIO

input_file = gzip.open("data/scop/pdb_seq.fasta.gz", "rt")
for line in input_file:
    if line.startswith(">"):
        pdbid = line[1:7].upper()
    elif pdbid in scop.index:
        row = scop.loc[pdbid]
        scop.loc[pdbid, "sequence"] = line.strip()[row.begin:row.end]
# remove unused columns and rows without a sequence
scop = scop.drop(["begin", "end", "chain"], axis=1)

scop = scop[scop.sequence.str.len() > 0]

scop.to_csv("processed/scop.csv.gz", index=False)

data = pandas.read_csv("processed/wheat.csv")

grid = seaborn.lmplot(x="compactness", y="length", data=data)
grid.savefig("results/figure.png", dpi=250, bbox_inches="tight")
from scipy import stats

r = stats.linregress(data.compactness, data.length)
r.slope
r.intercept
r.value
label=f"$R^2$ = {r.rvalue ** 2:.3f}"

grid.ax.plot(data.compactness, r.slope * data.compactness + r.intercept, "r", label=label)
grid.ax.legend()

grid.savefig("results/figure.png", dpi=250, bbox_inches="tight")

y = data.compactness
x = data.drop(["compactness", "variety"], axis=1)
from sklearn.linear_model import LinearRegression

lr = LinearRegression()
lr.fit(x, y)
lr.predict(x[:5])
lr.coef_
lr.intercept_

lr.score(x, y)
from sklearn.linear_model import Ridge, Lasso, ElasticNet

Ridge().fit(x, y).score(x, y)
Lasso().fit(x, y).score(x, y)
ElasticNet().fit(x, y).score(x, y)
# Classification
# Examples of classification models

# Logistic regression

y = data.variety
x = data.drop("variety", axis=1)
from sklearn.linear_model import LogisticRegression

lr = LogisticRegression(multi_class="ovr")
lr.fit(x, y)
lr.predict(x[:10])
lr.predict_proba(x[:10])
lr.score(x, y)
# limit model complexity
lr.set_params(C=0.1)
predicted = lr.fit(x, y).predict(x)
from sklearn import metrics

report = metrics.classification_report(y, predicted)
print(report)

data = pandas.read_pickle("processed/heart.pkl.gz")

x = data.drop("died", axis=1)
y = data.died

x.sex = x.sex.cat.codes
from sklearn.neighbors import KNeighborsClassifier

knn = KNeighborsClassifier()
predicted = knn.fit(x, y).predict(x)
from sklearn import pipeline, preprocessing

scaler = preprocessing.MinMaxScaler()
p = pipeline.make_pipeline(scaler, knn)

predicted = p.fit(x, y).predict(x)
knn.set_params(n_neighbors=3)
knn.set_params(weights="distance")
# Decision tree
# Categorical attributes
data = pandas.read_pickle("processed/mushroom.pkl.gz")

encoded = data.apply(lambda x: x.cat.codes)

y = encoded.kind
x = encoded.drop("kind", axis=1)

data = pandas.read_pickle("processed/mushroom.pkl.gz")

encoded = data.apply(lambda x: x.cat.codes)

y = encoded.kind
x = encoded.drop("kind", axis=1)
from sklearn.tree import DecisionTreeClassifier

dt = DecisionTreeClassifier(max_depth=2)
predicted = dt.fit(x, y).predict(x)
from sklearn import tree

text = tree.export_text(dt, feature_names=x.columns.tolist())
print(text)
x = pandas.get_dummies(data.iloc[:, 1:])
# increase model complexity
dt.set_params(max_depth=3)
predicted = dt.fit(x, y).predict(x)

data = pandas.read_csv("processed/wheat.csv")

# randomly remove 5% of values
sample = data.sample(frac=0.05)
data.loc[sample.index, "compactness"] = numpy.nan

# convert to a binary problem
y = data.variety.apply(lambda x: 1 if x < 2 else 0)

x = data.drop("variety", axis=1)
from sklearn import linear_model, preprocessing, impute, pipeline

classifier = linear_model.LogisticRegression()
scaler = preprocessing.RobustScaler()
imputer = impute.SimpleImputer(strategy="median")

p = pipeline.make_pipeline(imputer, scaler, classifier)
import numpy
rng = numpy.random.RandomState(31)

classifier.set_params(random_state=rng)

from sklearn import model_selection

model_selection.cross_val_score(p, x, y)
SEED = 127

cv = model_selection.StratifiedKFold(10, shuffle=True, random_state=SEED)
scores = model_selection.cross_val_score(p, x, y, scoring="f1", cv=cv)
scores.mean()
list(cv.split(x, y))[0]
# confidence interval
from scipy import stats

mu, sigma, n = scores.mean(), scores.std(), len(scores)
print(stats.t.interval(0.95, loc=mu, scale=sigma / math.sqrt(n), df=n - 1))

# stats.norm.interval(0.95, loc=mu, scale=sigma / math.sqrt(n)) # (if n >= 30)

# Measures of model quality
from sklearn import metrics

names = "precision recall f1 balanced_accuracy roc_auc accuracy".split()
scores = model_selection.cross_validate(p, x, y, scoring=names, cv=cv)

scores = pandas.DataFrame(scores)

grid = seaborn.catplot(x="variable", y="value", kind="box", data=scores.melt())
grid.set_xticklabels(rotation=45)
grid.savefig("results/figure.png", dpi=250, bbox_inches="tight")
# custom scorers
s = metrics.make_scorer(metrics.log_loss, greater_is_better=False, needs_proba=True)
model_selection.cross_val_score(p, x, y, scoring=s, cv=cv).mean()
x_train, x_test, y_train, y_test = model_selection.train_test_split(
    x, y, test_size=0.3, shuffle=True, random_state=SEED)
p.fit(x_train, y_train)
metrics.confusion_matrix(y_test, p.predict(x_test))
display = metrics.plot_precision_recall_curve(p, x_test, y_test)
display.figure_.savefig("results/figure.png", dpi=250, bbox_inches="tight")

display = metrics.plot_roc_curve(p, x_test, y_test)
display = metrics.plot_confusion_matrix(p, x_test, y_test)

# Learning curves
size, train, test = model_selection.learning_curve(p, x, y, scoring="f1", cv=cv,
    train_sizes=numpy.linspace(0.25, 1, 6), shuffle=True, random_state=SEED)
frame = pandas.DataFrame(train)
frame["size"] = size
frame["set"] = "train"
frame = frame.melt(id_vars=["size", "set"], var_name="fold", value_name="score")

frame2 = pandas.DataFrame(test)
...

data = frame.append(frame2, ignore_index=True)
grid = seaborn.relplot(x="size", y="score", hue="set", kind="line", data=data)
grid.set(xticks=size, xlim=(min(size), max(size)))
grid.savefig("results/figure.png", dpi=250, bbox_inches="tight")

# Grid search and nested CV
parameters = {
    "logisticregression__C": [0.01, 0.1, 1, 10, 100, 1000],
    "logisticregression__penalty": ["l1", "l2"],
    "logisticregression__class_weight": [None, "balanced"],
    "simpleimputer__strategy": ["mean", "median"]
}

classifier.set_params(solver="liblinear", max_iter=1000)
grid = model_selection.GridSearchCV(p, parameters, scoring="f1", cv=cv)
grid.fit(x, y)
grid.best_params_
grid.best_score_
grid.best_estimator_.score(x, y)

columns = "mean_test_score std_test_score params".split()
pandas.DataFrame(grid.cv_results_)[columns].sort_values(columns[0])
# outer folds: 60 vs 10 (70 instances of minority class)
outer = model_selection.StratifiedKFold(7, shuffle=True, random_state=SEED)
# inner folds: 50 vs 10 (60 instances of minority class)
inner = model_selection.StratifiedKFold(6, shuffle=True, random_state=SEED + 1)

grid = model_selection.GridSearchCV(p, parameters, scoring="f1", cv=inner)
scores = model_selection.cross_val_score(grid, x, y, scoring="f1", cv=outer, n_jobs=7)
scores.mean()

# SVM model training
data = pandas.read_pickle("processed/heart.pkl.gz")

y = data.died
x = data.drop("died", axis=1)
x.sex = x.sex.cat.codes
from sklearn.svm import SVC

svc = SVC(kernel="linear", class_weight="balanced")

svc.fit(x, y)
svc.predict_proba(x)  # not computed by default
from sklearn import pipeline, preprocessing
p = pipeline.make_pipeline(preprocessing.StandardScaler(), svc)
from sklearn import model_selection

cv = model_selection.StratifiedKFold(10, shuffle=True, random_state=127)
scores = model_selection.cross_val_score(p, x, y, scoring="f1", cv=cv)
svc.set_params(kernel="poly", degree=3)
svc.set_params(kernel="rbf", gamma=1 / x.shape[1])

# RF model training
data = pandas.read_csv("processed/wheat.csv")

x = data.iloc[:, :-1]
y = data.variety
from sklearn.ensemble import RandomForestClassifier

rf = RandomForestClassifier(100, max_depth=4, random_state=31)
scores = model_selection.cross_val_score(rf, x, y, scoring="f1_macro", cv=cv)
scores.mean()
rng = numpy.random.RandomState(31)
rf = RandomForestClassifier(100, max_depth=4, random_state=rng)
rf = RandomForestClassifier(100, max_depth=4, max_features=3, max_samples=0.5,
    bootstrap=False, criterion="entropy", class_weight="balance")

# Gradient boosted trees
from sklearn.ensemble import GradientBoostingClassifier

gb = GradientBoostingClassifier(n_estimators=100, max_depth=4, random_state=31)
model_selection.cross_val_score(gb, x, y, scoring="f1_macro", cv=cv).mean()
gb = GradientBoostingClassifier(n_estimators=10, max_depth=1, max_features=3,
    subsample=0.8, learning_rate=0.2, min_impurity_decrease=0.05)

# Stacking and voting
from sklearn.ensemble import StackingClassifier

estimators = list(zip("SVM RF ET GB", [p, rf, et, gb]))

s = StackingClassifier(estimators, final_estimator=None)
model_selection.cross_val_score(s, x, y, scoring="f1_macro", cv=cv).mean()
from sklearn.ensemble import VotingClassifier

SEEDS = [s for s in rng.randint(0, 10**9, 5)]
estimators = [(str(s), RandomForestClassifier(random_state=s)) for s in SEEDS]

v = VotingClassifier(estimators, voting="soft")
model_selection.cross_val_score(s, x, y, scoring="f1_macro", cv=cv).mean()

# Categorical attributes
data = pandas.read_pickle("processed/heart.pkl.gz")
y = data.died
x = data.drop("died", axis=1)
x.ejection = pandas.cut(data.ejection, [10,20,40,60,80])

categorical = x.columns[x.dtypes == "category"]
numeric = x.columns.difference(categorical)
from sklearn import impute
from sklearn.compose import ColumnTransformer

imputer_num = impute.KNNImputer(weights="distance")
imputer_cat = impute.SimpleImputer(strategy="most_frequent")

scaler = preprocessing.MinMaxScaler()
encoder = preprocessing.OrdinalEncoder()

transformer = ColumnTransformer([
    ("PN", pipeline.make_pipeline(imputer_num, scaler), numeric),
    ("PC", pipeline.make_pipeline(imputer_cat, encoder), categorical)])
outer_cv = model_selection.StratifiedKFold(6, shuffle=True, random_state=CV_SEED)

classifier = GradientBoostingClassifier()
p = pipeline.make_pipeline(transformer, classifier)
model_selection.cross_val_score(p, x, y, cv=outer_cv, scoring="f1").mean()

# Simple feature selection
from sklearn import feature_selection

selector = feature_selection.SelectKBest(k=5)
selector = feature_selection.SelectPercentile(feature_selection.mutual_info_classif)
inner_cv = model_selection.StratifiedKFold(8, shuffle=True, random_state=CV_SEED+1)

parameters = dict(n_estimators=[10, 20, 30], max_depth=[1,2,3,4])
grid_search = model_selection.GridSearchCV(classifier, parameters, cv=inner_cv, scoring="f1")

p = pipeline.make_pipeline(transformer, selector, grid_search)
model_selection.cross_val_score(p, x, y, cv=outer_cv, scoring="f1").mean()
# train final model
p.fit(x, y)

# Iterative feature selection
selection_cv = model_selection.StratifiedKFold(5, shuffle=True, random_state=CV_SEED+2)

# selection + nested CV
selector = feature_selection.RFECV(classifier, cv=selection_cv, scoring="f1", n_jobs=5)
p = pipeline.make_pipeline(transformer, selector, grid_search)
model_selection.cross_val_score(p, x, y, cv=outer_cv, scoring="f1").mean()
# triple nested CV
selector = feature_selection.RFECV(grid_search, cv=selection_cv, scoring="f1", n_jobs=5)
p = pipeline.make_pipeline(transformer, selector)
model_selection.cross_val_score(p, x, y, cv=outer_cv, scoring="f1").mean()
# override fit to store the feature importances
class MyGridSearchCV(model_selection.GridSearchCV):
    def fit(self, x, y=None, **fit_params):
        super(model_selection.GridSearchCV, self).fit(x, y, **fit_params)
        self.feature_importances_ = self.best_estimator_.feature_importances_
        return self

# Weights of linear models
data = pandas.read_pickle("processed/heart.pkl.gz")

y = data.died
x = data.drop("died", axis=1)
x.sex = x.sex.cat.codes
from sklearn.svm import LinearSVC

svc = LinearSVC(penalty="l1", dual=False, random_state=31)

from sklearn import preprocessing
svc.fit(preprocessing.scale(x), y)

svc.coef_

pandas.DataFrame(svc.coef_, columns=x.columns).T
from sklearn.linear_model import LinearRegression

lr = LogisticRegression(penalty="l1", solver="liblinear", random_state=31)
lr = LogisticRegression(penalty="l2", solver="liblinear", random_state=31)

# Impurity-based importance
data = pandas.read_csv("processed/wheat.csv")

x = data.iloc[:, :-1]
y = data.variety
from sklearn.ensemble import RandomForestClassifier

rf = RandomForestClassifier(100, max_depth=4, random_state=31)
rf.fit(x, y)
rf.feature_importances_
# plot importance
fig, p = plt.subplots()
order = rf.feature_importances_.argsort()
p.barh(x.columns[order], rf.feature_importances_[order])
fig.savefig("results/figure.png", dpi=250, bbox_inches="tight")
from sklearn import inspection
result = inspection.permutation_importance(rf, x, y, scoring="f1_macro", random_state=127)

# plot importance mean and stddev
fig, p = plt.subplots()
order = result.importances_mean.argsort()
p.barh(x.columns[order], result.importances_mean[order], xerr=result.importances_std[order])
fig.savefig("results/figure.png", dpi=250, bbox_inches="tight")

# SHapley Additive exPlanations (SHAP)
# TreeExplainer, KernelExplainer, DeepExplainer, GradientExplainer

import shap

rf = RandomForestClassifier(100, max_depth=4, random_state=31).fit(x, y)
explainer = shap.Explainer(rf)
explanation = explainer(x)

shap_values = explanation.values[:, :, 0]
expected = explanation.base_values[:, 0]
explanation = shap.Explanation(shap_values, expected, data=x.values, feature_names=x.columns)
shap.plots.bar(explanation, show=False)
plt.savefig("results/figure.png", dpi=250, bbox_inches="tight")
plt.close("all")
shap.plots.beeswarm(explanation, show=False)
shap.plots.heatmap(explanation, max_display=10, show=False)

# Contribution per instance
shap.plots.waterfall(explanation[0], show=False)
shap.plots.waterfall(explanation[100], show=False)
from sklearn import metrics
metrics.confusion_matrix(y, rf.predict(x))

x[y != rf.predict(x)]

shap.plots.waterfall(explanation[19], show=False)
shap.plots.waterfall(explanation[23], show=False)
shap.plots.waterfall(explanation[135], show=False)
[(i, p.round(2)) for i, p in enumerate(rf.predict_proba(x)) if abs(p[0] - sum(p[1:])) < 0.2]

shap.plots.waterfall(explanation[69], show=False)
shap.plots.force(expected[100], shap_values[100], x.columns, matplotlib=True, show=False)

# Partial dependence
fig, p = plt.subplots()
#inspection.DisplayPartialDependece.from_estimator(...) for sklearn >= 1.0
inspection.plot_partial_dependence(rf, x, target=1, ax=p,
    features=["area", "length", "compactness"])
fig.savefig("results/figure.png", dpi=250, bbox_inches="tight")
fig, p = plt.subplots()
inspection.plot_partial_dependence(rf, x, target=3, ax=p, features=[("length", "compactness")])
fig.savefig("results/figure.png", dpi=250, bbox_inches="tight")
# legacy SHAP dependency
fig, p = plt.subplots()
shap.dependence_plot("area", shap_values, x, ax=p)
fig.savefig("results/figure.png", dpi=250, bbox_inches="tight")
# new SHAP dependency
shap.plots.scatter(explanation[:, "width"], color=explanation)
plt.savefig("results/figure.png", dpi=250, bbox_inches="tight")
plt.close("all")


# Continuous attributes
data = pandas.read_csv("processed/wheat.csv")
x = data.iloc[:, :-1]
from sklearn import cluster

kmeans = cluster.KMeans(3)
kmeans.fit(x)
labels = kmeans.labels_
from sklearn import metrics

metrics.cluster.adjusted_mutual_info_score(data.variety, labels)
from sklearn import preprocessing

x_scaled = preprocessing.scale(x)
labels = cluster.KMeans(3).fit(x_scaled).labels_

x_scaled = preprocessing.robust_scale(x)
x_scaled = preprocessing.normalize(x)

# Parameter optimisation
labels = cluster.KMeans(5).fit(x_scaled).labels_

metrics.silhouette_score(x_scaled, labels)
# set state of the random number generator
random_state = numpy.random.RandomState(511)

best_score = -1
best_labels = None

for k in range(2, 16):
    for i in range(10):
        kmeans = cluster.KMeans(k, random_state=random_state)
        labels = kmeans.fit(x_scaled).labels_
        score = metrics.silhouette_score(x_scaled, labels)

        if score > best_score:
            best_score = score
            best_labels = labels

# Categorical attributes
data = pandas.read_pickle("processed/mushroom.pkl.gz")

x = data.iloc[:, 1:]
x = x.apply(lambda x: x.cat.codes)
from scipy.spatial import distance
distance.hamming([0,0,0,1], [0,1,0,1])

# calculate all pairwise distances
distances = distance.pdist(x, "hamming")
matrix = distance.squareform(distances)
agg = cluster.AgglomerativeClustering(n_clusters=None, distance_threshold=0.5,
    linkage="average", affinity="precomputed")
agg_labels = agg.fit(matrix).labels_
optics_labels = cluster.OPTICS(metric="hamming").fit(x).labels_
# Dimensionality reduction
from sklearn import manifold

embedding = manifold.MDS(random_state=511)
reduced = embedding.fit_transform(x_scaled)

a, b = reduced.T
import matplotlib.pyplot as plt
import seaborn

def plot(original, labels):
    fig, p = plt.subplots()
    seaborn.scatterplot(x=a, y=b, hue=labels, style=original, palette="deep", ax=p)
    fig.savefig("results/figure.png", dpi=250, bbox_inches="tight")
a, b = manifold.TSNE(random_state=511).fit_transform(x).T

a, b = manifold.TSNE(perplexity=15, random_state=511).fit_transform(x).T
a, b = manifold.TSNE(perplexity=15, n_iter=5000, random_state=511).fit_transform(x).T

# Model training
data = pandas.read_pickle("processed/heart.pkl.gz")
data.sex = data.sex.cat.codes

y = data.pop("died")
x = data
import keras
from keras.layers import Dense

def build_model():
    model = keras.Sequential([
        keras.Input(shape=(12,)),
        Dense(12, activation="relu"),
        Dense(1, activation="sigmoid")
    ])

    model.compile(loss="binary_crossentropy", optimizer="Adam")
    return model

nn = build_model()
nn.fit(x, y, validation_split=0.2, epochs=3, batch_size=1)

nn.summary()

# Model evaluation
def build_model():
    ...
    model.compile(loss="binary_crossentropy", optimizer="Adam", metrics=["Precision", "Recall"])
...

nn.evaluate(x, y)
nn.predict(x)
# will extra pre-processing help?
from sklearn import preprocessing

x = preprocessing.scale(x)
from scikeras import wrappers
# bridge between tensorflow and sklearn
nn = wrappers.KerasClassifier(build_model, epochs=5, verbose=0)

nn.fit(x, y)
nn.predict(x)
nn.predict_proba(x)
# cross validation
from sklearn import model_selection
scores = model_selection.cross_val_score(nn, x, y, scoring="f1", cv=5)
scores.mean()

# Learning curves
nn = build_model()
history = nn.fit(x, y, validation_split=0.2, epochs=30, verbose=0)

data = pandas.DataFrame(history.history)

data["epoch"] = history.epoch
data = data.melt(id_vars="epoch", var_name="measure")

data["set"] = ["validation" if x else "training" for x in data.measure.str.startswith("val")]
data.measure = data.measure.apply(lambda x: x.split("_")[-1])

grid = seaborn.relplot(x="epoch", y="value", hue="set", col="measure", data=data)
grid.savefig("results/figure.png", dpi=250, bbox_inches="tight")

Setup
data = pandas.read_csv("processed/scop.csv.gz")

data.sequence.str.len().describe()
data = data[data.sequence.str.len() >= 50]

# select largest 6 families
top = data.family_name.value_counts()[:6].index
data = data[data.family_name.isin(top)]
from sklearn import preprocessing

y = preprocessing.LabelBinarizer().fit_transform(data.family_name)

# learn the vocabulary from sequences
alphabet = set("".join(data.sequence))
vocabulary = dict(zip(alphabet, range(1, len(alphabet) + 1)))
# convert sequences to lists of tokens
x = data.sequence.apply(lambda x: [vocabulary[i] for i in x])

# pad short / trim long sequences
MAX_LENGTH = 308
x = keras.utils.pad_sequences(x, MAX_LENGTH, truncating="post")

CNN
from keras.layers import Embedding, Conv1D, MaxPooling1D
from keras.layers import Dropout, Flatten, Dense

TOKENS = len(vocabulary) + 1
CLASSES = 6
DIMENSIONS = 8
UNITS = 32
SIZE = 4
DROPOUT_RATE = 0.2

def build_cnn():
    model = keras.Sequential([
        keras.Input(shape=(MAX_LENGTH,)),
        Embedding(TOKENS, DIMENSIONS, mask_zero=False),

        Conv1D(UNITS, SIZE, activation="relu"),
        MaxPooling1D(SIZE),
        Dropout(DROPOUT_RATE),

        Conv1D(UNITS, SIZE, activation="relu"),
        MaxPooling1D(SIZE),
        Dropout(DROPOUT_RATE),

        Flatten(),
        Dense(UNITS, activation="relu"),
        Dense(CLASSES, activation="softmax")
    ])

    model.compile(loss="categorical_crossentropy", optimizer="Adam", metrics=["accuracy"])
    return model
# Model training
nn = build_cnn()

nn.fit(x, y, validation_split=0.2, epochs=30)

nn.summary()

LSTM
from keras.layers import LSTM, Bidirectional

def build_lstm():
    model = keras.Sequential([
        keras.Input(shape=(MAX_LENGTH,)),
        Embedding(TOKENS, DIMENSIONS, mask_zero=True),

        Bidirectional(LSTM(UNITS, return_sequences=True)),
        Bidirectional(LSTM(UNITS // 2)),

        Dense(UNITS, activation="relu"),
        Dropout(DROPOUT_RATE),

        Dense(CLASSES, activation="softmax")
    ])

    model.compile(loss="categorical_crossentropy", optimizer="Adam", metrics=["accuracy"])
    return model

# GRU + Simple RNN
from keras.layers import GRU, SimpleRNN

def build_gru():
    model = keras.Sequential([
        keras.Input(shape=(MAX_LENGTH,)),
        Embedding(TOKENS, DIMENSIONS, mask_zero=True),

        GRU(UNITS, return_sequences=True),
        SimpleRNN(UNITS // 2),

        Dense(UNITS, activation="relu"),
        Dropout(DROPOUT_RATE),

        Dense(CLASSES, activation="softmax")
    ])

    model.compile(loss="categorical_crossentropy", optimizer="Adam", metrics=["accuracy"])
    return model
# What to do next?
# use scikit-learn wrapper
# (cross-validation, grid search)

# use shap for interpretation
# (DeepExplainer, GradientExplainer)

# Transformer architecture
# Encoder-Decoder

# Tay et al. 2022, "Efficient Transformers: A Survey"

# Implementation
from keras.layers import MultiHeadAttention, LayerNormalization

class TransformerEncoder(keras.layers.Layer):
    def __init__(self, size, dimensions, heads):
        super().__init__()
        self.attention = MultiHeadAttention(heads, dimensions)
        self.dense = keras.Sequential([
            Dense(size, activation="relu"),
            Dense(dimensions),
            Dropout(0.1)
        ])
        self.norm1 = LayerNormalization()
        self.norm2 = LayerNormalization()

    def call(self, inputs):
        outputs = self.norm1(self.attention(inputs, inputs) + inputs)
        return self.norm2(self.dense(outputs) + outputs)
class TokenAndPositionEmbedding(keras.layers.Layer):
    def __init__(self, tokens, length, dimensions):
        super().__init__()
        self.tokens = Embedding(tokens, dimensions)
        self.positions = Embedding(length, dimensions)
        self.indices = numpy.arange(0, length)

    def call(self, x):
        return self.tokens(x) + self.positions(self.indices)

Transformer
from keras.layers import GlobalAveragePooling1D

HEADS = 2
DIMENSIONS = 16

def build_transformer():
    model = keras.Sequential([
        keras.Input(shape=(MAX_LENGTH,)),
        TokenAndPositionEmbedding(TOKENS, MAX_LENGTH, DIMENSIONS),

        TransformerEncoder(UNITS, DIMENSIONS, HEADS),
        TransformerEncoder(UNITS, DIMENSIONS, HEADS),
        GlobalAveragePooling1D(),

        Dense(UNITS, activation="relu"),
        Dropout(DROPOUT_RATE),

        Dense(CLASSES, activation="softmax")
    ])

    model.compile(loss="categorical_crossentropy", optimizer="Adam", metrics=["accuracy"])
    return model
