# Optimized version of the Von Mises Mixture Model using EM
from scipy.special import i0

class VonMisesMixture:

    def __init__(self, n_components, max_iter=100, tol=1e-6):
        self.n_components = n_components  # Number of von Mises components
        self.max_iter = max_iter  # Maximum iterations for EM
        self.tol = tol  # Convergence threshold
        self.weights_ = None  # Mixture weights
        self.means_ = None  # Mean directions (mu)
        self.kappas_ = None  # Concentration parameters

    def initialize_parameters(self, data):
        """ Initialize parameters using random selection """
        self.weights_ = np.ones(self.n_components) / self.n_components  # Equal weights initially
        self.means_ = np.random.choice(data, self.n_components, replace=False)  # Random initial means
        self.kappas_ = np.full(self.n_components, 1.0)  # Initial kappas set to 1

    def von_mises_pdf(self, x, mu, kappa):
        """ Compute the von Mises PDF efficiently """
        return np.exp(kappa * np.cos(x - mu)) / (2 * np.pi * i0(kappa))

    def e_step(self, data):
        """ Expectation step: compute responsibilities """
        n = len(data)
        responsibilities = np.zeros((n, self.n_components))

        for k in range(self.n_components):
            responsibilities[:, k] = self.weights_[k] * self.von_mises_pdf(data, self.means_[k], self.kappas_[k])
        
        responsibilities /= responsibilities.sum(axis=1, keepdims=True)  # Normalize
        return responsibilities

    def m_step(self, data, responsibilities):
        """ Maximization step: update parameters """
        n_k = responsibilities.sum(axis=0)  # Effective number of points for each component

        # Update weights
        self.weights_ = n_k / len(data)

        # Update means (mu) using circular mean
        for k in range(self.n_components):
            C_k = np.sum(responsibilities[:, k] * np.cos(data)) / n_k[k]
            S_k = np.sum(responsibilities[:, k] * np.sin(data)) / n_k[k]
            self.means_[k] = np.arctan2(S_k, C_k)

            # Update kappa using an approximation
            R_k = np.sqrt(C_k**2 + S_k**2)
            if R_k < 0.53:
                self.kappas_[k] = 2 * R_k + R_k**3 + (5 / 6) * R_k**5
            elif 0.53 <= R_k < 0.85:
                self.kappas_[k] = -0.4 + 1.39 * R_k + 0.43 / (1 - R_k)
            else:
                self.kappas_[k] = 1 / (2 * (1 - R_k))

    def fit(self, data):
        """ Fit the von Mises mixture model using EM """
        self.initialize_parameters(data)
        prev_likelihood = -np.inf

        for iteration in range(self.max_iter):
            responsibilities = self.e_step(data)
            self.m_step(data, responsibilities)

            # Compute log-likelihood
            log_likelihood = np.sum(np.log(np.sum([self.weights_[k] * self.von_mises_pdf(data, self.means_[k], self.kappas_[k])
                                                   for k in range(self.n_components)], axis=0)))

            # Check for convergence
            if np.abs(log_likelihood - prev_likelihood) < self.tol:
                break
            prev_likelihood = log_likelihood

        # determine cluster ids based on cluster with max likelihood
        cluster_ids = np.argmax(responsibilities,axis=1)
        
        return cluster_ids, log_likelihood
    
    def predict(self, data):
        """Predict cluster ids"""
        
        responsibilities = self.e_step(data)
        
        # determine cluster ids based on cluster with max likelihood
        cluster_ids = np.argmax(responsibilities,axis=1)
        
        # Compute log-likelihood
        log_likelihood = np.sum(np.log(np.sum([self.weights_[k] * self.von_mises_pdf(data, self.means_[k], self.kappas_[k])
                                                   for k in range(self.n_components)], axis=0)))
        return cluster_ids, log_likelihood
    
    def pdf(self, data):
        """Compute the pdf of the mixture on the data"""
        pdf_mixture = np.zeros_like(data)
        for k in range(self.n_components):
            pdf_mixture += self.weights_[k] * self.von_mises_pdf(data, self.means_[k], self.kappas_[k])
        return pdf_mixture
