# Fair Federated Learning under Domain Skew with Local Consistency and Domain Diversity

## Abstract

Federated learning (FL) has emerged as a new paradigm for privacy-preserving collaborative training. Under domain skew, the current FL approaches are biased and face two fairness problems:

1. **Parameter Update Conflict**: Data disparity among clients leads to varying parameter importance and inconsistent update directions. These two disparities cause important parameters to potentially be overwhelmed by unimportant ones of dominant updates. It consequently results in significant performance decreases for lower-performing clients.

2. **Model Aggregation Bias**: Existing FL approaches introduce unfair weight allocation and neglect domain diversity. It leads to biased model convergence objectives and distinct performance among domains.

We discover a pronounced **directional update consistency** in Federated Learning and propose a novel framework to tackle the above issues. 

### Key Contributions:
- **Selective Update Discarding**: Leveraging the discovered characteristic, we selectively discard unimportant parameter updates to prevent updates from clients with lower performance from being overwhelmed by unimportant parameters, resulting in fairer generalization performance.
- **Fair Aggregation Objective**: We propose a fair aggregation objective to prevent global model bias towards some domains, ensuring that the global model continuously aligns with an unbiased model.

The proposed method is generic and can be combined with other existing FL methods to enhance fairness.

Comprehensive experiments on **Digits** and **Office-Caltech** demonstrate the high fairness and performance of our method.

