# FROM frotaur/base-ml:latest
FROM zigapusnik/review_and_assessment_bn_inference:latest

USER root

COPY ./requirements.txt /reviewAndAssessment/implementations
RUN pip3 install -r requirements.txt

COPY . /reviewAndAssessment/implementations

CMD ["/bin/bash", "-c","tail -f /dev/null"]

# copy paste from Vassilis (=frotaur)