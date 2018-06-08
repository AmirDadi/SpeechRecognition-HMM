from hmmlearn import hmm
import numpy as np
from torch.autograd import Variable
import torch


class ANNHMM(hmm.GMMHMM):
    def __init__(self, ann_model, n_components=1, n_mix=1,
                 startprob_prior=1.0, transmat_prior=1.0,
                 covariance_type='diag', covars_prior=1e-2,
                 algorithm="viterbi", random_state=None,
                 n_iter=10, tol=1e-2, verbose=False,
                 params="stmcw", init_params="stmcw"):
        hmm.GMMHMM.__init__(self, n_components, n_mix,
                            startprob_prior=startprob_prior,
                            transmat_prior=transmat_prior,
                            covariance_type=covariance_type,
                            covars_prior=covars_prior,
                            algorithm=algorithm, random_state=random_state,
                            n_iter=n_iter, tol=tol, verbose=verbose,
                            params=params, init_params=init_params)
        self.Ann_model = ann_model
        self.is_gmm = True

    def _initialize_sufficient_statistics(self):
        stats = super(ANNHMM, self)._initialize_sufficient_statistics()
        stats['data'] = []
        stats['labels'] = []
        return stats

    def _accumulate_sufficient_statistics(self, stats, X, framelogprob,
                                          posteriors, fwdlattice, bwdlattic):
        if self.is_gmm:
            super(ANNHMM, self)._accumulate_sufficient_statistics(stats, X, framelogprob,
                                                                  posteriors, fwdlattice, bwdlattic)
        else:
            stats['data'].append(X)
            stats['labels'].append(np.argmax(framelogprob, axis=1))
        return self

    def _compute_log_likelihood(self, X):
        if self.is_gmm:
            x = super(ANNHMM, self)._compute_log_likelihood(X)
            return x
        else:
            # for sample in X:
            x = self.Ann_model(
                Variable(torch.from_numpy(X))
            ).data.numpy() - np.log(self.startprob_)
            # print(x)
            return x.astype(np.float64)
            # x = np.array(likelihoods, dtype='float32')
            # return x

    def fit(self, X, lengths=None):
        self.is_gmm = True
        super(ANNHMM, self).fit(X, lengths)

        print('gmm done')
        print(self.monitor_)
        for i, j in ANNHMM.iter_from_X_lengths(X, lengths):
            framelogprob = self._compute_log_likelihood(X[i:j])
            self.Ann_model.train(X[i:j],
                                 np.argmax(framelogprob, axis=1))
        print('first train done')

        self.is_gmm = False
        for i in range(20):
            print(i)
            super(ANNHMM, self).fit(X, lengths)
            print(self.monitor_)

        print(self.monitor_)

    def _do_mstep(self, stats):
        if self.is_gmm:
            super(ANNHMM, self)._do_mstep(stats)
        else:
            # for label, sample in zip(stats['data'], stats['label']):
            print('do m step')
            print(self.Ann_model.train(np.concatenate(stats['data']),
                                 np.concatenate(stats['labels'])))

    @staticmethod
    def iter_from_X_lengths(X, lengths):
        if lengths is None:
            yield 0, len(X)
        else:
            n_samples = X.shape[0]
            end = np.cumsum(lengths).astype(np.int32)
            start = end - lengths
            if end[-1] > n_samples:
                raise ValueError("more than {0:d} samples in lengths array {1!s}"
                                 .format(n_samples, lengths))

            for i in range(len(lengths)):
                yield start[i], end[i]