# YooChose-dataset-analysis
This repository deals with the analysis of youchoose dataset and compares some of the papers publised on Recomender sytems that have used it as a dataset. We try to find some patterns in data and try to draw some conclusions out of it. 

# Here are the links to papers I have mentioned in my Data analysis Code. 
1. Gated Graph Based RNN for Recomendations : Session-based Recommendation with Graph Neural Networks by Shu Wu et al
2. GRU4REC : SESSION-BASED RECOMMENDATIONS WITH RECURRENT NEURAL NETWORKS by Balazas Hidasi et al
3. STAMP : STAMP: Short-Term Attention/Memory Priority Model for Session-based Recommendation by Qiao et al
4. NARM : Neural Attentive Session-based Recommendation by Jing Li et al

Youchoose Dataset can be found on kaggle : https://www.kaggle.com/chadgostopp/recsys-challenge-2015

#The abhove analysis has been done on processed data, the code for processing the youchoose dataset is in **preprocessing.py**.

### Here are some conclusions that can be drawn from this data analysis:

-Some items are not visited frequently so we can just eliminate them from our prediction or decrease there probability so that we get more results that are more likely to be      clicked.
-Most sessions are similar in one way or other to one or more previously existing sessions hence Graph Neural Networks that can model this sort of similarity perform very well.
-It is repeatedly seen that the last ItemIDs have more number of clicks in every session, which highlights the fact that the last click is very Important in determining the next click.
-The general attention calculation will fail to understand the complexity of small click sequences and hence the incorporation of first click is necessary in order to take in effect the  idea of the user, therefore I propose that to increase the efficiency of models, when the number of sessions are less than a threshold, First click should also be used for doing the prediction.
-As it is evident that making sense of these points is out of question for human beings and that most of these sessions have repletion's in part with other sequences, the RNN based models are best suitable for this session based prediction .
