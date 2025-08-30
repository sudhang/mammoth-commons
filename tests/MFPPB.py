from abc import ABCMeta, abstractmethod
from copy import deepcopy
import numpy as np
import sklearn
from sklearn.base import is_classifier, ClassifierMixin, is_regressor
from sklearn.ensemble import BaseEnsemble

# from sklearn.ensemble.forest import BaseForest
# from sklearn.externals
import six
from typing import Literal
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.metrics import r2_score
from sklearn.tree._classes import BaseDecisionTree, DTYPE, DecisionTreeClassifier
from sklearn.utils.validation import (
    has_fit_parameter,
    check_is_fitted,
    check_array,
    check_X_y,
    check_random_state,
)
from pymoo.mcdm.pseudo_weights import PseudoWeights

__all__ = ["Multi_Fair"]


class BaseWeightBoosting(six.with_metaclass(ABCMeta, BaseEnsemble)):
    """Base class for Boosting.

    Warning: This class should not be used directly. Use derived classes
    instead.
    """

    @abstractmethod
    def __init__(
        self,
        estimator=None,
        n_estimators=50,
        estimator_params=tuple(),
        learning_rate=1.0,
        random_state=None,
    ):
        super(BaseWeightBoosting, self).__init__(
            estimator=estimator,
            n_estimators=n_estimators,
            estimator_params=estimator_params,
        )

        self.W_pos = [0 for i in range(5)]
        self.W_neg = [0 for i in range(5)]
        self.W_dp = [0 for i in range(5)]
        self.W_fp = [0 for i in range(5)]
        self.W_dn = [0 for i in range(5)]
        self.W_fn = [0 for i in range(5)]

        self.theta = n_estimators
        self.performance = []
        self.objective = []
        self.objective_opti = []
        self.final_objective = []
        self.fairloss = []
        self.max_sensi = []
        self.learning_rate = learning_rate
        self.random_state = random_state
        self.tuning_learners = []
        self.tuning_optimals = []
        self.sol = {}
        self.ob1, self.ob2, self.ob3, self.ob4 = [], [], [], []

    def fit(self, X, y, sample_weight=None):
        """Build a boosted classifier/regressor from the training set (X, y).

        Parameters
        ----------
        X : {array-like, sparse matrix} of shape = [n_samples, n_features]
            The training input samples. Sparse matrix can be CSC, CSR, COO,
            DOK, or LIL. COO, DOK, and LIL are converted to CSR. The dtype is
            forced to DTYPE from tree._tree if the base classifier of this
            ensemble weighted boosting classifier is a tree or forest.

        y : array-like of shape = [n_samples]
            The target values (class labels in classification, real numbers in
            regression).

        sample_weight : array-like of shape = [n_samples], optional
            Sample weights. If None, the sample weights are initialized to
            1 / n_samples.

        Returns
        -------
        self : object
            Returns self.
        """
        # Check parameters
        self.weight_list = []
        self.costs_list = []
        if self.learning_rate <= 0:
            raise ValueError("learning_rate must be greater than zero")

        if self.estimator is None or isinstance(self.estimator, BaseDecisionTree):
            dtype = DTYPE
            accept_sparse = "csc"
        else:
            dtype = None
            accept_sparse = ["csr", "csc"]

        X, y = check_X_y(
            X, y, accept_sparse=accept_sparse, dtype=dtype, y_numeric=is_regressor(self)
        )

        if sample_weight is None:
            # Initialize weights to 1 / n_samples
            sample_weight = np.empty(X.shape[0], dtype=np.float64)
            sample_weight[:] = 1.0 / X.shape[0]
        else:
            sample_weight = check_array(sample_weight, ensure_2d=False)
            # Normalize existing weights
            sample_weight = sample_weight / sample_weight.sum(dtype=np.float64)

            # Check that the sample weights sum is positive
            if sample_weight.sum() <= 0:
                raise ValueError(
                    "Attempting to fit with a non-positive "
                    "weighted number of samples."
                )

        self.predictions_array = np.zeros([X.shape[0], 2])

        # Check parameters
        self._validate_estimator()

        if self.debug:
            self.conf_scores = []

        # Clear any previous fit results
        self.estimators_ = []

        self.estimator_alphas_ = np.zeros(self.n_estimators + 1, dtype=np.float64)
        self.estimator_fairness_ = np.ones(self.n_estimators + 1, dtype=np.float64)

        random_state = check_random_state(self.random_state)
        if self.debug:
            print("Begin Debug")

        old_weights_sum = np.sum(sample_weight)
        wg, tp, tn, pp, npp, pn, npn = self.calculate_weights(X, y, sample_weight)
        wgs = [str(v) for v in wg]

        if self.debug:
            self.weight_list.append("init" + "," + str(0) + "," + ",".join(wgs))

        flag, iboost, best_theta = 0, -1, 0
        T = self.n_estimators
        self.ob = []
        self.fairobs = []
        while iboost < T:
            # Boosting step
            iboost += 1
            (
                sample_weight,
                alpha,
                error,
                fairness,
                eq_odds,
                balanced_loss,
                cumulative_loss,
            ) = self._boost(iboost, X, y, sample_weight, random_state)

            # Early termination
            if sample_weight is None:
                break

            self.ob.append([cumulative_loss, balanced_loss, max(fairness)])
            self.fairobs.append(fairness)
            if error == 0.5:
                print("Bad Estimator")
                break

            new_sample_weight = np.sum(sample_weight)
            multiplier = old_weights_sum / new_sample_weight

            # Stop if the sum of sample weights has become non-positive
            if new_sample_weight <= 0:
                break

            if iboost < self.n_estimators - 1:
                # Normalize
                sample_weight *= multiplier

            if self.debug:
                self.weight_list.append(
                    str(iboost) + "," + str(alpha) + "," + ",".join(wgs)
                )
                wg, tp, tn, pp, npp, pn, npn = self.calculate_weights(
                    X, y, sample_weight
                )
                wgs = [str(v) for v in wg]

                for i in range(len(tp)):
                    self.W_pos[i] += tp[i] / self.n_estimators
                    self.W_neg[i] += tn[i] / self.n_estimators
                    self.W_dp[i] += pp[i] / self.n_estimators
                    self.W_fp[i] += npp[i] / self.n_estimators
                    self.W_dn[i] += pn[i] / self.n_estimators
                    self.W_fn[i] += npn[i] / self.n_estimators

            old_weights_sum = np.sum(sample_weight)

        self.ob = np.array(self.ob)
        self.fairobs = np.array(self.fairobs)
        # self.update_theta(criteria='all')
        # if best_theta==0:
        # def is_pareto(costs, maximise=False):
        #     """
        #     :param costs: An (n_points, n_costs) array
        #     :maximise: boolean. True for maximising, False for minimising
        #     :return: A (n_points, ) boolean array, indicating whether each point is Pareto efficient
        #     """
        #     is_efficient = np.ones(costs.shape[0], dtype = bool)
        #     for i, c in enumerate(costs):
        #         if is_efficient[i]:
        #             if maximise:
        #                 is_efficient[is_efficient] = np.any(costs[is_efficient]>=c, axis=1)  # Remove dominated points
        #             else:
        #                 is_efficient[is_efficient] = np.any(costs[is_efficient]<=c, axis=1)  # Remove dominated points
        #     return is_efficient
        # best_theta=0
        # self.ob=np.array(self.ob)
        # if self.pareto==False:
        #     self.PF={i:self.ob[i] for i in range(len(self.ob))}
        #     F=np.array([self.ob[o] for o in range(len(self.ob))])
        # else:
        #     pf=is_pareto(self.ob)
        #     self.PF={i:self.ob[i] for i in range(len(pf)) if pf[i]==True}
        #     F=np.array(list(self.PF.values()))

        # weights = self.preference  ##Preference Weights
        # if weights==None:
        #     weights=[0.33,0.34,0.33]

        # best_theta, self.pseudo_weights = PseudoWeights(weights).do(F, return_pseudo_weights=True)

        # if self.preference==None:
        #     sum_W=[sum((1-self.pseudo_weights[w])*F[w]) for w in range(len(self.PF))]
        #     best_theta=sum_W.index(min(sum_W))

        # self.theta = list(self.PF.keys())[best_theta] + 1

        if self.debug:
            print("best partial ensemble at round: " + str(self.theta))
        # self.estimators_ = self.estimators_[:self.theta  ]
        # self.estimator_alphas_ = self.estimator_alphas_[:self.theta  ]

        if self.debug:
            print("total #weak learners = " + str(len(self.estimators_)))
            self.get_confidence_scores(X)

        return self

    def get_weights_over_iterations(
        self,
    ):
        return self.weight_list[self.theta]

    def get_confidence_scores(self, X):
        self.conf_scores = self.decision_function(X)

    def get_initial_weights(self):
        return self.weight_list[0]

    def get_weights(
        self,
    ):
        weights = []
        for i in range(len(self.W_pos)):
            weights.append(self.W_pos[i])
            weights.append(self.W_neg[i])
            weights.append(self.W_dp[i])
            weights.append(self.W_fp[i])
            weights.append(self.W_dn[i])
            weights.append(self.W_fn[i])

        return weights

    @abstractmethod
    def _boost(self, iboost, X, y, sample_weight, random_state):
        """Implement a single boost.

        Warning: This method needs to be overridden by subclasses.

        Parameters
        ----------
        iboost : int
            The index of the current boost iteration.

        X : {array-like, sparse matrix} of shape = [n_samples, n_features]
            The training input samples. Sparse matrix can be CSC, CSR, COO,
            DOK, or LIL. COO, DOK, and LIL are converted to CSR.

        y : array-like of shape = [n_samples]
            The target values (class labels).

        sample_weight : array-like of shape = [n_samples]
            The current sample weights.

        random_state : numpy.RandomState
            The current random number generator

        Returns
        -------
        sample_weight : array-like of shape = [n_samples] or None
            The reweighted sample weights.
            If None then boosting has terminated early.

        estimator_weight : float
            The weight for the current boost.
            If None then boosting has terminated early.

        error : float
            The classification error for the current boost.
            If None then boosting has terminated early.
        """

    def staged_score(self, X, y, sample_weight=None):
        """Return staged scores for X, y.

        This generator method yields the ensemble score after each iteration of
        boosting and therefore allows monitoring, such as to determine the
        score on a test set after each boost.

        Parameters
        ----------
        X : {array-like, sparse matrix} of shape = [n_samples, n_features]
            The training input samples. Sparse matrix can be CSC, CSR, COO,
            DOK, or LIL. DOK and LIL are converted to CSR.

        y : array-like, shape = [n_samples]
            Labels for X.

        sample_weight : array-like, shape = [n_samples], optional
            Sample weights.

        Returns
        -------
        z : float
        """
        for y_pred in self.staged_predict(X):
            if is_classifier(self):
                yield accuracy_score(y, y_pred, sample_weight=sample_weight)
            else:
                yield r2_score(y, y_pred, sample_weight=sample_weight)

    @property
    def feature_importances_(self):
        """Return the feature importances (the higher, the more important the
           feature).

        Returns
        -------
        feature_importances_ : array, shape = [n_features]
        """
        if self.estimators_ is None or len(self.estimators_) == 0:
            raise ValueError(
                "Estimator not fitted, " "call `fit` before `feature_importances_`."
            )

        try:
            norm = self.estimator_alphas_.sum()
            return (
                sum(
                    weight * clf.feature_importances_
                    for weight, clf in zip(self.estimator_alphas_, self.estimators_)
                )
                / norm
            )

        except AttributeError:
            raise AttributeError(
                "Unable to compute feature importances "
                "since base_estimator does not have a "
                "feature_importances_ attribute"
            )

    def _validate_X_predict(self, X):
        """Ensure that X is in the proper format"""
        if self.estimator is None or isinstance(self.estimator, BaseDecisionTree):
            X = check_array(X, accept_sparse="csr", dtype=DTYPE)

        else:
            X = check_array(X, accept_sparse=["csr", "csc", "coo"])

        return X

    def calculate_weights(self, data, labels, sample_weight):
        protected_positive = [0 for i in self.saValue]
        non_protected_positive = [0 for i in self.saValue]

        protected_negative = [0 for i in self.saValue]
        non_protected_negative = [0 for i in self.saValue]

        for idx, val in enumerate(data):
            for i in range(len((self.saValue))):
                con = True
                if isinstance(self.saValue[self.sensitives[i]], list):
                    if (
                        self.saIndex[idx][i] <= self.saValue[self.sensitives[i]][0]
                        or self.saIndex[idx][i] >= self.saValue[self.sensitives[i]][1]
                    ):
                        con = True
                    else:
                        con = False
                elif isinstance(self.saValue[self.sensitives[i]], float):
                    if self.saIndex[idx][i] <= self.saValue[self.sensitives[i]]:
                        con = True
                    elif self.saIndex[idx][i] > self.saValue[self.sensitives[i]]:
                        con = False
                elif isinstance(self.saValue[self.sensitives[i]], int):
                    if self.saIndex[idx][i] == self.saValue[self.sensitives[i]]:
                        con = True
                    elif self.saIndex[idx][i] != self.saValue[self.sensitives[i]]:
                        con = False
                if con == True:
                    if labels[idx] == 1:
                        protected_positive[i] += sample_weight[idx]
                    else:
                        protected_negative[i] += sample_weight[idx]

                elif con == False:
                    if labels[idx] == 1:
                        non_protected_positive[i] += sample_weight[idx]
                    else:
                        non_protected_negative[i] += sample_weight[idx]

            tp = [
                protected_positive[i] + non_protected_positive[i]
                for i in range(len(self.saValue))
            ]
            tn = [
                protected_negative[i] + non_protected_negative[i]
                for i in range(len(self.saValue))
            ]
            pp = [protected_positive[i] for i in range(len(self.saValue))]
            npp = [non_protected_positive[i] for i in range(len(self.saValue))]
            pn = [protected_negative[i] for i in range(len(self.saValue))]
            npn = [non_protected_negative[i] for i in range(len(self.saValue))]

            tot = []
            for i in range(len((self.saValue))):
                tot.append(tp[i])
                tot.append(tn[i])
                tot.append(pp[i])
                tot.append(npp[i])
                tot.append(pn[i])
                tot.append(npn[i])

        return tot, tp, tn, pp, npp, pn, npn


def _samme_proba(estimator, n_classes, X):
    """Calculate algorithm 4, step 2, equation c) of Zhu et al [1].

    References
    ----------
    .. [1] J. Zhu, H. Zou, S. Rosset, T. Hastie, "Multi-class AdaBoost", 2009.

    """
    proba = estimator.predict_proba(X)

    # Displace zero probabilities so the log is defined.
    # Also fix negative elements which may occur with
    # negative sample weights.
    proba[proba < np.finfo(proba.dtype).eps] = np.finfo(proba.dtype).eps
    log_proba = np.log(proba)

    return (n_classes - 1) * (
        log_proba - (1.0 / n_classes) * log_proba.sum(axis=1)[:, np.newaxis]
    )


class Multi_Fair(BaseWeightBoosting, ClassifierMixin):
    def __init__(
        self,
        estimator=None,
        n_estimators=50,
        learning_rate=1.0,
        algorithm="SAMME",
        random_state=None,
        saIndex: np.ndarray = None,
        saValue: dict = {},
        debug=False,
        X_test=None,
        y_test=None,
        preference=None,
        pareto=False,
        pos_class=None,
        # sensitives=None
    ):  # ,protected_attr=['Race','Sex']):
        super(Multi_Fair, self).__init__(
            estimator=estimator,
            n_estimators=n_estimators,
            learning_rate=learning_rate,
            random_state=random_state,
        )

        self.preference = preference  ########Initialization of Preference Weight vector
        self.pareto = pareto
        self.saIndex = saIndex
        self.saValue = saValue
        self.algorithm = algorithm

        self.cost_protected_positive = [1 for i in self.saValue]
        self.cost_non_protected_positive = [1 for i in self.saValue]
        self.cost_protected_negative = [1 for i in self.saValue]
        self.cost_non_protected_negative = [1 for i in self.saValue]
        self.estimators_ = None
        self.estimator_alphas_ = None
        self.classes_ = None
        self.n_classes_ = None
        self.costs = []
        self.PF = {}
        self.sensitives = list(self.saValue.keys())
        """
        if sensitives!=None:
            self.prot_attr=sensitives
            if len(self.saValue)!=len(self.prot_attr):
                raise Exception(
                    "Number of protected attributes do not match with the number of protected values"
                )
        # self.PF_f={}
        """
        self.debug = debug

        self.X_test = X_test
        self.y_test = y_test
        self.pseudo = None

        if pos_class != None:
            self.pos_class = pos_class
        else:
            self.pos_class = 1

    def fit(self, X, y, sample_weight=None):
        """Build a boosted classifier from the training set (X, y).

        Parameters
        ----------
        X : {array-like, sparse matrix} of shape = [n_samples, n_features]
            The training input samples. Sparse matrix can be CSC, CSR, COO,
            DOK, or LIL. DOK and LIL are converted to CSR.

        y : array-like of shape = [n_samples]
            The target values (class labels).

        sample_weight : array-like of shape = [n_samples], optional
            Sample weights. If None, the sample weights are initialized to
            ``1 / n_samples``.

        Returns
        -------
        self : object
            Returns self.
        """
        # Check that algorithm is supported
        if self.algorithm not in ("SAMME", "SAMME.R"):
            raise ValueError("algorithm %s is not supported" % self.algorithm)

        # Fit
        return super(Multi_Fair, self).fit(X, y, sample_weight)

    def _validate_estimator(self):
        """Check the estimator and set the base_estimator_ attribute."""
        super(Multi_Fair, self)._validate_estimator(
            default=DecisionTreeClassifier(max_depth=1)
        )

        #  SAMME-R requires predict_proba-enabled base estimators
        if self.algorithm == "SAMME.R":
            if not hasattr(self.estimator_, "predict_proba"):
                raise TypeError(
                    "AccumFairAdaCost with algorithm='SAMME.R' requires "
                    "that the weak learner supports the calculation of class "
                    "probabilities with a predict_proba method.\n"
                    "Please change the base estimator or set "
                    "algorithm='SAMME' instead."
                )
        if not has_fit_parameter(self.estimator_, "sample_weight"):
            raise ValueError(
                "%s doesn't support sample_weight." % self.estimator_.__class__.__name__
            )

    def _boost(self, iboost, X, y, sample_weight, random_state):
        return self._boost_discrete(iboost, X, y, sample_weight, random_state)

    def calculate_fairness(self, data, labels, predictions):
        # TODO: this function needs optimization by employing the numpy structures
        tp_protected = [0 for i in self.saValue]
        tn_protected = [0 for i in self.saValue]
        fp_protected = [0 for i in self.saValue]
        fn_protected = [0 for i in self.saValue]

        tp_non_protected = [0 for i in self.saValue]
        tn_non_protected = [0 for i in self.saValue]
        fp_non_protected = [0 for i in self.saValue]
        fn_non_protected = [0 for i in self.saValue]

        for idx, val in enumerate(data):
            for i in range(len(self.saValue)):
                # con=True
                if isinstance(self.saValue[self.sensitives[i]], list):
                    if (
                        self.saIndex[idx][i] <= self.saValue[self.sensitives[i]][0]
                        or self.saIndex[idx][i] >= self.saValue[self.sensitives[i]][1]
                    ):
                        con = True
                    else:
                        con = False
                elif isinstance(self.saValue[self.sensitives[i]], float):
                    if self.saIndex[idx][i] <= self.saValue[self.sensitives[i]]:
                        con = True
                    elif self.saIndex[idx][i] > self.saValue[self.sensitives[i]]:
                        con = False
                else:
                    if self.saIndex[idx][i] == self.saValue[self.sensitives[i]]:
                        con = True
                    elif self.saIndex[idx][i] != self.saValue[self.sensitives[i]]:
                        con = False
                if con == True:  # protrcted population
                    if labels[idx] == predictions[idx]:  # correctly classified
                        if labels[idx] == 1:
                            tp_protected[i] += 1
                        else:
                            tn_protected[i] += 1
                    # misclassified
                    else:
                        if labels[idx] == 1:
                            fn_protected[i] += 1
                        else:
                            fp_protected[i] += 1

                elif con == False:
                    # correctly classified
                    if labels[idx] == predictions[idx]:
                        if labels[idx] == 1:
                            tp_non_protected[i] += 1
                        else:
                            tn_non_protected[i] += 1
                    # misclassified
                    else:
                        if labels[idx] == 1:
                            fn_non_protected[i] += 1
                        else:
                            fp_non_protected[i] += 1

        # tpr_protected = [0 for i in self.saIndex]
        # tnr_protected = [0 for i in self.saIndex]
        # tpr_non_protected = [0 for i in self.saIndex]
        # tnr_non_protected = [0 for i in self.saIndex]
        cost = []
        fair_cost, eq_odds = [], []
        lists = ""
        for i in range(len(self.saValue)):
            tpr_protected = tp_protected[i] / (tp_protected[i] + fn_protected[i])
            tpr_non_protected = tp_non_protected[i] / (
                tp_non_protected[i] + fn_non_protected[i]
            )

            tnr_protected = tn_protected[i] / (tn_protected[i] + fp_protected[i])
            tnr_non_protected = tn_non_protected[i] / (
                tn_non_protected[i] + fp_non_protected[i]
            )

            diff_tpr = tpr_non_protected - tpr_protected
            diff_tnr = tnr_non_protected - tnr_protected
            fair_cost.append(max(abs(diff_tpr), abs(diff_tnr)))
            eq_odds.append(abs(diff_tpr) + abs(diff_tnr))
            cost.append(str(diff_tpr))
            cost.append(str(diff_tnr))

            if diff_tpr > 0:
                self.cost_protected_positive[i] = 1 + diff_tpr
            elif diff_tpr < 0:
                self.cost_non_protected_positive[i] = 1 + abs(diff_tpr)
            else:
                self.cost_protected_positive[i] = 1
                self.cost_non_protected_positive[i] = 1

            if diff_tnr > 0:
                self.cost_protected_negative[i] = 1 + diff_tnr
            elif diff_tnr < 0:
                self.cost_non_protected_negative[i] = 1 + abs(diff_tnr)
            else:
                self.cost_protected_negative[i] = 1
                self.cost_non_protected_positive[i] = 1
            lists = (
                lists
                + str(self.cost_protected_positive[i])
                + ","
                + str(self.cost_non_protected_positive[i])
                + ","
                + str(self.cost_protected_negative[i])
                + ","
                + str(self.cost_non_protected_negative[i])
                + ","
            )

        self.costs_list.append(lists)

        # print str(self.cost_protected_positive) + "," + str(self.cost_non_protected_positive)  + "," + str(self.cost_protected_negative)  + "," + str(self.cost_non_protected_negative)
        self.costs.append(",".join(cost))

        return fair_cost, eq_odds

    def update_theta(
        self,
        criteria: Literal["all", "fairness"] = "all",
        preference=[0.33, 0.34, 0.33],
    ):
        def is_pareto(costs, maximise=False):
            """
            :param costs: An (n_points, n_costs) array
            :maximise: boolean. True for maximising, False for minimising
            :return: A (n_points, ) boolean array, indicating whether each point is Pareto efficient
            """
            eps = 0.001
            is_efficient = np.ones(costs.shape[0], dtype=bool)
            for i, c in enumerate(costs):
                if is_efficient[i]:
                    if maximise:
                        is_efficient[is_efficient] = np.any(
                            costs[is_efficient] >= c, axis=1
                        )  # Remove dominated points
                    else:
                        is_efficient[is_efficient] = np.any(
                            costs[is_efficient] <= c, axis=1
                        )  # Remove dominated points
            return is_efficient

        self.preference = preference
        best_theta = 0
        if criteria.lower() == "fairness":
            objective = deepcopy(self.fairobs)
        else:
            objective = deepcopy(self.ob)
        # objective=np.round(objective,2)
        if self.pareto == False:
            PF = {i: objective[i] for i in range(len(objective))}
            F = np.array([objective[o] for o in range(len(objective))])
            self.PF = PF
        else:
            pf = is_pareto(objective)
            PF = {i: objective[i] for i in range(len(pf)) if pf[i] == True}
            F = np.array(list(PF.values()))
            self.PF = PF

        weights = self.preference  ##Preference Weights

        best_theta, pseudo_weights = PseudoWeights(weights).do(
            F, return_pseudo_weights=True
        )

        if self.preference == None:
            sum_W = [sum((1 - pseudo_weights[w]) * F[w]) for w in range(len(PF))]
            best_theta = sum_W.index(min(sum_W))

        self.theta = list(PF.keys())[best_theta] + 1
        self.pseudo = pseudo_weights

    """
    def see_pareto(self):
        import plotly.graph_objs as go
        from plotly.subplots import make_subplots
        def plot2d(x,y,axis_names):
            pass
        def plot3d(x = [1, 2, 3, 4, 5],y = [2, 3, 1, 4, 5],z = [3, 1, 2, 5, 4],weights = [1,2,3,4,5],criteria='all',axis_names=['X','Y','Z']):
            # Create the figure
            hidden_data=[str(np.around(weights[i],2)) for i in range(len(weights))]
            fig = make_subplots(rows=1, cols=1, specs=[[{'type': 'scatter3d'}]])
            
            # Add 3D scatter plot
            fig.add_trace(go.Scatter3d(x=x, y=y, z=z, mode='markers', marker=dict(color='blue'), 
                                       name='Points',text=hidden_data, customdata=hidden_data ), row=1, col=1)
            
            # Define callback function for selection event
            def update_point(trace, points, selector):
                selected_points = [(trace.x[i], trace.y[i], trace.z[i]) for i in points.point_inds]
                selected_hidden_data = [trace.customdata[i] for i in points.point_inds]
                print("Selected Points:", selected_points,selected_hidden_data)
            
            # Assign callback function to plot
            fig.data[0].on_selection(update_point)
            
            # Update layout
            fig.update_layout(title=criteria+'-objectives Pareto Plot - Select Points to Return Vector',
                             scene=dict(
                                        xaxis_title='X:'+axis_names[0],
                                        yaxis_title='Y:'+ axis_names[1],
                                        zaxis_title='Z:'+ axis_names[2]
                                    ),
                                    width=1000,  # Set the width of the plot
                                    height=700 )
            
            # Show plot
            fig.show()
        
        self.update_theta(criteria='all')
        plot3d(x=self.PF[:,0],y=self.PF[:,1],z=self.PF[:,2], weights=self.pseudo,
               axis_names=['Acc.','Balanc. Acc', 'MMM-fair'])
        self.update_theta(criteria='fairness')
        if len(self.prot_attr)>2:
            plot3d(x=self.PF[:,0],y=self.PF[:,1],z=self.PF[:,2], weights=self.pseudo,
                   criteria='Fairness',axis_names=self.prot_attr)
        elif len(self.prot_attr)==2:
                plot3d(x=self.PF[:,0],y=self.PF[:,1],z=np.zeros_like(self.PF[:,1]), weights=self.pseudo,
                   criteria='Fairness',axis_names=self.prot_attr+'')
    
        else:
            print('Not a Multi-fair')
    def measure_fairness_for_visualization(self, data, labels, predictions):

        tp_protected = [0 for i in self.saIndex]
        tn_protected = [0 for i in self.saIndex]
        fp_protected = [0 for i in self.saIndex]
        fn_protected = [0 for i in self.saIndex]

        tp_non_protected = [0 for i in self.saIndex]
        tn_non_protected = [0 for i in self.saIndex]
        fp_non_protected = [0 for i in self.saIndex]
        fn_non_protected = [0 for i in self.saIndex]
        
        for idx, val in enumerate(data):                            
            for i in range(len(self.saIndex)):
              con=True
              if isinstance(self.saValue[i], list):
                    if val[self.saIndex[i]] <= self.saValue[i][0] or val[self.saIndex[i]] >= self.saValue[i][1]:
                        con=True
                    else:
                        con=False
              elif isinstance(self.saValue[i], float):
                    if val[self.saIndex[i]] <= self.saValue[i]:
                        con=True
                    elif val[self.saIndex[i]] > self.saValue[i]:
                        con=False
              elif isinstance(self.saValue[i], int):
                    if val[self.saIndex[i]] == self.saValue[i]:
                        con=True
                    elif val[self.saIndex[i]] != self.saValue[i]:
                        con=False        
              if con==True:    # protrcted population
                if labels[idx] == predictions[idx]:   # correctly classified
                    if labels[idx] == 1:
                        tp_protected[i] +=1
                    else:
                        tn_protected[i] +=1
                #misclassified
                else:
                    if labels[idx] == 1:
                        fn_protected[i] +=1
                    else:
                        fp_protected[i] +=1

              elif con==False:
                # correctly classified
                if labels[idx] == predictions[idx]:
                    if labels[idx] == 1:
                        tp_non_protected[i] +=1
                    else:
                        tn_non_protected[i] +=1
                # misclassified
                else:
                    if labels[idx] == 1:
                        fn_non_protected[i] +=1
                    else:
                        fp_non_protected[i] +=1

        fair_cost=[]
        for i in range(len(self.saIndex)):
            tpr_protected = tp_protected[i]/(tp_protected[i] + fn_protected[i])
            tpr_non_protected = tp_non_protected[i]/(tp_non_protected[i] + fn_non_protected[i])
            
            tnr_protected = tn_protected[i]/(tn_protected[i] + fp_protected[i])
            tnr_non_protected = tn_non_protected[i]/(tn_non_protected[i] + fp_non_protected[i])
            diff_tpr = tpr_non_protected - tpr_protected
            diff_tnr = tnr_non_protected - tnr_protected
            fair_cost.append(abs(diff_tpr) + abs(diff_tnr))
        j=fair_cost.index(max(fair_cost))
        max_sen=self.saIndex[j]
        return max(fair_cost),max_sen
    """

    def _boost_discrete(self, iboost, X, y, sample_weight, random_state):
        """Implement a single boost using the SAMME discrete algorithm."""
        estimator = self._make_estimator(random_state=random_state)
        estimator.fit(X, y, sample_weight=sample_weight)
        y_predict = estimator.predict(X)
        proba = estimator.predict_proba(X)

        if iboost == 0:
            self.classes_ = getattr(estimator, "classes_", None)
            self.n_classes_ = len(self.classes_)
        # n_classes = self.n_classes_
        incorrect = y_predict != y

        # Error fraction
        estimator_error = np.mean(np.average(incorrect, weights=sample_weight, axis=0))

        # Stop if classification is perfect
        if estimator_error <= 0:
            return sample_weight, 1.0, 0.0

        n_classes = self.n_classes_

        # Stop if the error is at least as bad as random guessing
        if estimator_error >= 1.0 - (1.0 / n_classes):
            self.estimators_.pop(-1)
            if len(self.estimators_) == 0:
                raise ValueError(
                    "BaseClassifier in AdaBoostClassifier "
                    "ensemble is worse than random, ensemble "
                    "can not be fit."
                )
            return None, None, None

        # Boost weight using multi-class AdaBoost SAMME alg
        alpha = self.learning_rate * (
            np.log((1.0 - estimator_error) / estimator_error) + np.log(n_classes - 1.0)
        )

        # incorrect = y_predict != y
        self.estimator_alphas_[iboost] = alpha
        self.predictions_array += (y_predict == self.classes_[:, np.newaxis]).T * alpha

        # Error fraction
        # estimator_error = np.mean(np.average(incorrect, weights=sample_weight, axis=0))

        if iboost != 0:
            # cumulative_balanced_error = 1 - sklearn.metrics.balanced_accuracy_score(y, self.predict(X))
            fairness, eq_ods = self.calculate_fairness(
                X, y, self.classes_.take(np.argmax(self.predictions_array, axis=1))
            )
            # cumulative_error = 1 - sklearn.metrics.accuracy_score(y, self.predict(X))
        else:
            # cumulative_error = estimator_error
            # cumulative_balanced_error = 1 - sklearn.metrics.balanced_accuracy_score(y, y_predict)
            fairness = [1 for i in self.saValue]
            eq_ods = [1 for i in self.saValue]

        """
        For fast training -to reduce actual runtime the loss functions are measured using tp, fp, fn, and tn.
        In principle the functions works same as the formal functions defined in the literature.

        """
        tn, fp, fn, tp = confusion_matrix(
            y,
            self.classes_.take(np.argmax(self.predictions_array, axis=1), axis=0),
            # labels=[0, 1],
        ).ravel()
        TPR = (float(tp)) / (tp + fn)
        TNR = (float(tn)) / (tn + fp)

        cumulative_balanced_loss = abs(TPR - TNR)
        cumulative_loss = 1 - (float(tp) + float(tn)) / (tp + tn + fp + fn)

        if not iboost == self.n_estimators - 1:
            for idx, row in enumerate(sample_weight):
                if y[idx] == self.pos_class and y_predict[idx] != self.pos_class:
                    cost_p = [1 for i in self.saValue]
                    for i in range(len(self.saValue)):
                        con = True
                        if isinstance(self.saValue[self.sensitives[i]], list):
                            if (
                                self.saIndex[idx][i]
                                <= self.saValue[self.sensitives[i]][0]
                                or self.saIndex[idx][i]
                                >= self.saValue[self.sensitives[i]][1]
                            ):
                                con = True
                            else:
                                con = False
                        elif isinstance(self.saValue[self.sensitives[i]], float):
                            if self.saIndex[idx][i] <= self.saValue[self.sensitives[i]]:
                                con = True
                            elif (
                                self.saIndex[idx][i] > self.saValue[self.sensitives[i]]
                            ):
                                con = False
                        elif isinstance(self.saValue[self.sensitives[i]], int):
                            if self.saIndex[idx][i] == self.saValue[self.sensitives[i]]:
                                con = True
                            elif (
                                self.saIndex[idx][i] != self.saValue[self.sensitives[i]]
                            ):
                                con = False
                        if con == True:
                            cost_p[i] = self.cost_protected_positive[i]
                        elif con == False:
                            cost_p[i] = self.cost_non_protected_positive[i]

                    sample_weight[idx] *= max(cost_p) * np.exp(
                        alpha * max(proba[idx][0], proba[idx][1])
                    )

                elif y[idx] != self.pos_class and y_predict[idx] == self.pos_class:
                    cost_n = [1 for i in self.saValue]
                    for i in range(len(self.saValue)):
                        con = True
                        if isinstance(self.saValue[self.sensitives[i]], list):
                            if (
                                self.saIndex[idx][i]
                                <= self.saValue[self.sensitives[i]][0]
                                or self.saIndex[idx][i]
                                >= self.saValue[self.sensitives[i]][1]
                            ):
                                con = True
                            else:
                                con = False
                        elif isinstance(self.saValue[self.sensitives[i]], float):
                            if self.saIndex[idx][i] <= self.saValue[self.sensitives[i]]:
                                con = True
                            elif (
                                self.saIndex[idx][i] > self.saValue[self.sensitives[i]]
                            ):
                                con = False
                        elif isinstance(self.saValue[self.sensitives[i]], int):
                            if self.saIndex[idx][i] == self.saValue[self.sensitives[i]]:
                                con = True
                            elif (
                                self.saIndex[idx][i] != self.saValue[self.sensitives[i]]
                            ):
                                con = False
                        if con == True:
                            cost_n[i] = self.cost_protected_negative[i]
                        elif con == False:
                            cost_n[i] = self.cost_non_protected_negative[i]

                    sample_weight[idx] *= max(cost_n) * np.exp(
                        alpha * max(proba[idx][0], proba[idx][1])
                    )

        if self.debug:
            y_predict = self.predict(X)
            incorrect = y_predict != y
            train_error = np.mean(np.average(incorrect, axis=0))
            train_bal_error = 1 - sklearn.metrics.balanced_accuracy_score(y, y_predict)
            train_fairness, ms = self.measure_fairness_for_visualization(
                X, y, y_predict
            )

            test_error = 0
            test_bal_error = 0
            test_fairness = 0
            if self.X_test is not None:
                y_predict = self.predict(self.X_test)
                incorrect = y_predict != self.y_test
                test_error = np.mean(np.average(incorrect, axis=0))
                test_bal_error = 1 - sklearn.metrics.balanced_accuracy_score(
                    self.y_test, y_predict
                )
                test_fairness = self.measure_fairness_for_visualization(
                    self.X_test, self.y_test, y_predict
                )

            self.max_sensi.append(ms)
            self.performance.append(
                str(iboost)
                + ","
                + str(train_error)
                + ", "
                + str(train_bal_error)
                + ", "
                + str(train_fairness)
                + ","
                + str(test_error)
                + ", "
                + str(test_bal_error)
                + ", "
                + str(test_fairness)
            )
            # print ('iter- '+str(iboost)+',')# + "," + str(train_error) + ", " + str(train_bal_error) + ", " + str(train_fairness) + ","+ str(test_error) + ", " + str(test_bal_error)+ ", " + str(test_fairness))

        return (
            sample_weight,
            alpha,
            estimator_error,
            fairness,
            eq_ods,
            cumulative_balanced_loss,
            cumulative_loss,
        )

    def get_performance_over_iterations(self):
        return self.performance

    #
    def get_objective(self):
        return self.objective

    def get_faircost(self):
        return self.fairloss, self.max_sensi

    #
    # def get_weights_over_iterations(self):
    #     return self.weight_list[self.theta]

    def predict(self, X):
        """Predict classes for X.

        The predicted class of an input sample is computed as the weighted mean
        prediction of the classifiers in the ensemble.

        Parameters
        ----------
        X : {array-like, sparse matrix} of shape = [n_samples, n_features]
            The training input samples. Sparse matrix can be CSC, CSR, COO,
            DOK, or LIL. DOK and LIL are converted to CSR.

        Returns
        -------
        y : array of shape = [n_samples]
            The predicted classes.
        """
        pred = self.decision_function(X)

        if self.n_classes_ == 2:
            return self.classes_.take(pred > 0, axis=0)

        return self.classes_.take(np.argmax(pred, axis=1), axis=0)

    def decision_function(self, X):
        """Compute the decision function of ``X``.
        Parameters
        ----------
        X : {array-like, sparse matrix} of shape = [n_samples, n_features]
            The training input samples. Sparse matrix can be CSC, CSR, COO,
            DOK, or LIL. DOK and LIL are converted to CSR.
        Returns
        -------
        score : array, shape = [n_samples, k]
            The decision function of the input samples. The order of
            outputs is the same of that of the `classes_` attribute.
            Binary classification is a special cases with ``k == 1``,
            otherwise ``k==n_classes``. For binary classification,
            values closer to -1 or 1 mean more like the first or second
            class in ``classes_``, respectively.
        """
        check_is_fitted(self, "n_classes_")
        X = self._validate_X_predict(X)

        n_classes = self.n_classes_
        classes = self.classes_[:, np.newaxis]

        pred = sum(
            (estimator.predict(X) == classes).T * w
            for estimator, w in zip(
                self.estimators_[: self.theta], self.estimator_alphas_[: self.theta]
            )
        )
        pred /= self.estimator_alphas_[: self.theta].sum()
        if n_classes == 2:
            pred[:, 0] *= -1
            return pred.sum(axis=1)
        return pred

    def predict_proba(self, X):
        """Predict class probabilities for X.

        The predicted class probabilities of an input sample is computed as
        the weighted mean predicted class probabilities of the classifiers
        in the ensemble.

        Parameters
        ----------
        X : {array-like, sparse matrix} of shape = [n_samples, n_features]
            The training input samples. Sparse matrix can be CSC, CSR, COO,
            DOK, or LIL. DOK and LIL are converted to CSR.

        Returns
        -------
        p : array of shape = [n_samples]
            The class probabilities of the input samples. The order of
            outputs is the same of that of the `classes_` attribute.
        """
        check_is_fitted(self, "n_classes_")

        n_classes = self.n_classes_
        X = self._validate_X_predict(X)

        if n_classes == 1:
            return np.ones((X.shape[0], 1))

        proba = sum(
            estimator.predict_proba(X) * w
            for estimator, w in zip(
                self.estimators_[: self.theta], self.estimator_alphas_[: self.theta]
            )
        )

        proba /= self.estimator_alphas_[: self.theta].sum()
        proba = np.exp((1.0 / (n_classes - 1)) * proba)
        normalizer = proba.sum(axis=1)[:, np.newaxis]
        normalizer[normalizer == 0.0] = 1.0
        proba /= normalizer

        return proba

    def predict_log_proba(self, X):
        """Predict class log-probabilities for X.

        The predicted class log-probabilities of an input sample is computed as
        the weighted mean predicted class log-probabilities of the classifiers
        in the ensemble.

        Parameters
        ----------
        X : {array-like, sparse matrix} of shape = [n_samples, n_features]
            The training input samples. Sparse matrix can be CSC, CSR, COO,
            DOK, or LIL. DOK and LIL are converted to CSR.

        Returns
        -------
        p : array of shape = [n_samples]
            The class probabilities of the input samples. The order of
            outputs is the same of that of the `classes_` attribute.
        """
        return np.log(self.predict_proba(X))
