# Deploying an end-to-end chicken disease model into cloud server using Flask and Docker with CI/CD pipeline
![deploy](https://github.com/hamzajakouk/chicken_disease_classification/workflows/Build%20and%20deploy/badge.svg)
![tests](https://github.com/hamzajakouk/chicken_disease_classification/workflows/Tests/badge.svg)
![releases](https://img.shields.io/github/v/release/hamzajakouk/chicken_disease_classification)

This project focuses on the prediction of chicken diseases using an `end-to-end deep learning` approach, with the integration of `MLOps`, `DVC pipeline` , and `Azure` for deployments. The `pipeline` encompasses various stages, beginning with `data preprocessing`  and `feature extraction` from chicken health records. The deep learning model is `trained` on the processed data to predict the occurrence of chicken diseases accurately. Throughout the project, `MLOps` practices are employed to `track` and `log` experiments, including `model artifacts`, parameters, and metrics. The resulting model is then integrated into a web application, which is `containerized` using `Docker`. The final step involves `deploying` the application, along with the trained model, as a Docker container on Azure. The deployment process leverages `Azure`'s capabilities for `CI/CD integration`, `automated tests`, and `releases`, ensuring a robust and efficient management of the deployed chicken disease prediction system.






