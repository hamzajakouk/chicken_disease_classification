# Deploying an end-to-end chicken disease model into cloud server using Flask and Docker with CI/CD pipeline

This project focuses on the prediction of chicken diseases using an `end-to-end deep learning` approach, with the integration of `MLOps`, `DVC pipeline` , and `Azure` for deployments. The `pipeline` encompasses various stages, beginning with `data preprocessing`  and `feature extraction` from chicken health records. The deep learning model is `trained` on the processed data to predict the occurrence of chicken diseases accurately. Throughout the project, `MLOps` practices are employed to `track` and `log` experiments, including `model artifacts`, parameters, and metrics. The resulting model is then integrated into a web application, which is `containerized` using `Docker`. The final step involves `deploying` the application, along with the trained model, as a Docker container on Azure. The deployment process leverages `Azure`'s capabilities for `CI/CD integration`, `automated tests`, and `releases`, ensuring a robust and efficient management of the deployed chicken disease prediction system.

## Author

- [@Hamza Jakouk](https://www.github.com/hamzajakouk)

## Languages and Tools

<div align="">
<a href="https://www.python.org" target="_blank" rel="noreferrer"><img src="https://raw.githubusercontent.com/devicons/devicon/master/icons/python/python-original.svg" alt="python" width="40" height="40"/></a>
<a href="https://www.tensorflow.org" target="_blank" rel="noreferrer"><img src="https://www.vectorlogo.zone/logos/tensorflow/tensorflow-icon.svg" alt="tensorflow" width="40" height="40"/></a>
<a href="https://www.docker.com/" target="_blank" rel="noreferrer"><img src="https://raw.githubusercontent.com/devicons/devicon/master/icons/docker/docker-original-wordmark.svg" alt="docker" width="40" height="40"/></a>
<a href="https://flask.palletsprojects.com/en/2.2.x/" target="_blank" rel="noreferrer"> <img src="https://banner2.cleanpng.com/20180704/sv/kisspng-flask-python-web-framework-bottle-microframework-django-5b3d0ba62504c0.3512153115307273341516.jpg" alt="flask" width="95" height="43"/></a>
<a href="https://github.com/features/actions" target="_blank" rel="noreferrer"> <img src="https://res.cloudinary.com/practicaldev/image/fetch/s--2mFgk66y--/c_limit,f_auto,fl_progressive,q_80,w_375/https://dev-to-uploads.s3.amazonaws.com/uploads/badge/badge_image/78/github-actions-runner-up-badge.png" alt="actions" width="52" height="49"/></a>
<a href="https://azure.microsoft.com/" target="_blank" rel="noreferrer"><img src="https://www.vectorlogo.zone/logos/microsoft_azure/microsoft_azure-icon.svg" alt="azure" width="40" height="40"/></a>
<a href="https://th.bing.com/th/id/R.82986e112ecefabbcfbaf960e2c8fb36?rik=SfIPu6t13UB87Q&pid=ImgRaw&r=0" target="_blank" rel="noreferrer"><img src="https://th.bing.com/th/id/R.82986e112ecefabbcfbaf960e2c8fb36?rik=SfIPu6t13UB87Q&pid=ImgRaw&r=0" alt="dvc" width="40" height="40"/></a>
</div>

## Chicken Disease in Azure - Demo

_**Link**: Will be updated. Please check the `Disclaimer` below the screenshot for more !!!_

| ![input](./images/heroku_in.PNG) |
|:--:|
| <b>Figure 1a: App demo - Audio input to app for predicting keyword from trained model artifact</b>|

| ![input](./images/heroku_out.PNG) |
|:--:|
| <b>Figure 1b: App demo - Predicted keyword with probability</b>|

_**Disclaimer:**_ <br>
_1. This app is just a demo and not for realtime usage. The main objective is to get ML models into production in terms of deployment and CI/CD, from MLOps paradigm_. <br>
_2. Additionally, due to some technical issues in the Heroku backend, the app currently crashes, so the Heroku app link is not provided as of now. It will be updated once the issues are solved and when the app is up and running_.
  



