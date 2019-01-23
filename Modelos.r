
library(randomForest)
library(caret)
library(sqldf)

df = read.csv('data/Archive/ContasReceber&Faturamento_24-10.csv', sep=',', fileEncoding='utf-8')

head(df, 3)

dim(df)

# removendo valores na
df = na.omit(df)

dim(df)

# removendo colunas que não serão usadas, que são:
# 'fant_shop_data', 'NrMes', 'date_type', 'Shopping', 'NmFantasia',
#  'GrupoAtividade', 'RamoAtividade', 'TipoAtividade'

df2 = df[, -c(1:8)]

dim(df2)

inad1 = df2[, -c(153,154)]
head(inad1)

# data set com idicação de 
# inadimplência para o próximo mês
inad_1 = df2[, -c(153,154)]
inad_1 = subset(inad_1, inad1 != -1)

# data set com idicação de 
# inadimplência 2 mês à frente
inad_2 = df2[, -c(154, 155)]
inad_2 = subset(inad_2, inad2 != -1)

# data set com idicação de 
# inadimplência 3 mês à frente
inad_3 = df2[, -c(153, 155)]
inad_3 = subset(inad_3, inad3 != -1)

dim(inad_1)

head(inad_1)

# gerando índices do conjunto de treino
train = sample (1: nrow(inad_1), nrow(inad_1) / 2)

length(train)

inad_1['inad1'] = as.factor(inad_1$inad1)

inad_1['inad'] = as.factor(inad_1$inad)

inad_1Modelo = randomForest(formula =  inad1 ~ ., data = inad_1, subset = train,
                            mtry = 50, ntree = 100)

inad_1Modelo

varImpPlot(inad_1Modelo,  
           sort = T,
           n.var=10,
           main="Importância das Variáveis: Top 10")

# Data frame da importância de cada variável
var.imp = data.frame(importance(inad_1Modelo,  
                                 type=2))
var.imp$Variables = row.names(var.imp)  
print(var.imp[order(var.imp$MeanDecreaseGini,decreasing = T),])

teste_inad_1 = inad_1[-train, ]

# Fazendo previsão num data set teste
teste_inad_1[ 'previsao'] = predict(inad_1Modelo , newdata = teste_inad_1)

head(teste_inad_1, 3)

# Matriz de Confusão

confusionMatrix(data = teste_inad_1$previsao,  
                reference = teste_inad_1$inad1,
                positive = '1')

# gerando índices do conjunto de treino
train2 = sample (1: nrow(inad_2), nrow(inad_2) / 2)

length(train2)

inad_2['inad2'] = as.factor(inad_2$inad2)

inad_2['inad'] = as.factor(inad_2$inad)

inad_2Modelo = randomForest(formula =  inad2 ~ ., data = inad_2, subset = train2,
                            mtry = 50, ntree = 100)

inad_2Modelo

varImpPlot(inad_2Modelo,  
           sort = T,
           n.var=10,
           main="Importância das Variáveis: Top 10")

teste_inad_2 = inad_2[-train2, ]

# Fazendo previsão num data set teste
teste_inad_2[ 'previsao'] = predict(inad_2Modelo , newdata = teste_inad_2)

# Matriz de Confusão

confusionMatrix(data = teste_inad_2$previsao,  
                reference = teste_inad_2$inad2,
                positive = '1')

# gerando índices do conjunto de treino
train3 = sample (1: nrow(inad_3), nrow(inad_3) / 2)

length(train3)

inad_3['inad3'] = as.factor(inad_3$inad3)
inad_3['inad'] = as.factor(inad_3$inad)

inad_3Modelo = randomForest(formula =  inad3 ~ ., data = inad_3, subset = train3,
                            mtry = 50, ntree = 100)

inad_3Modelo

varImpPlot(inad_3Modelo,  
           sort = T,
           n.var=10,
           main="Importância das Variáveis: Top 10")

teste_inad_3 = inad_3[-train3, ]

# Fazendo previsão num data set teste
teste_inad_3[ 'previsao'] = predict(inad_3Modelo , newdata = teste_inad_3)

confusionMatrix(data = teste_inad_3$previsao,  
                reference = teste_inad_3$inad3,
                positive = '1')

# modelo de 3 meses usando todas as variáveis
inad_3Modelo_td = randomForest(formula =  inad3 ~ ., data = inad_3, subset = train3,
                            mtry = 152, ntree = 500)

inad_3Modelo_td

varImpPlot(inad_3Modelo_td,  
           sort = T,
           n.var=10,
           main="Importância das Variáveis: Top 10")

# Fazendo previsão num data set teste
teste_inad_3[ 'previsao_td'] = predict(inad_3Modelo_td , newdata = teste_inad_3)

confusionMatrix(data = teste_inad_3$previsao_td,  
                reference = teste_inad_3$inad3,
                positive = '1')

df_201806 = subset(df, df$NrMes == 201806)

head(df_201806)

dim(df_201806)

# removendo as colunas
# inad1, inad2 e inad3
df_201806 = df_201809[,-c(161, 162, 163)]

head(df_201806, 3)

df_201806['inad'] = as.factor(df_201806$inad)

# Fazendo previsão usando os modelos 
# Random Forest já gerados. As colunas 
# 1 à 8 não são consideradas como input.
df_201806['pred_julho'] = predict(inad_1Modelo , newdata = df_201806[ , -c(1:8)])
df_201806['pred_agosto'] = predict(inad_2Modelo, newdata = df_201806[ , -c(1:8)])
df_201806['pred_setembro'] = predict(inad_3Modelo, newdata = df_201806[ , -c(1:8)])

head(df_201806[, c('NrMes', 'Shopping', 'NmFantasia','pred_julho', 'pred_agosto', 'pred_setembro')])

write.csv(df_201806[, c('NrMes', 'Shopping', 'NmFantasia','pred_julho', 'pred_agosto', 'pred_setembro')]
          ,'data/df_2018-06-modelo-random-forest.csv', sep=',', fileEncoding="utf-8", row.names=FALSE)

# listando empresas que 
# que obteve previsão de inadimplência
# nos 3 modelos
caso_juridico = sqldf('select fant_shop_data, Shopping, NmFantasia, 
                                pred_outubro, pred_novembro, pred_dezembro from df_201809
                        where pred_outubro = 1
                        and pred_novembro = 1
                        and pred_dezembro = 1
                        ')

dim(caso_juridico)

head(caso_juridico)

unique(caso_juridico$NmFantasia)

unique(caso_juridico$fant_shop_data)

glm_inad_1 = glm ( inad1 ~., data = inad_1 , 
                  family = binomial, subset = train)

summary(glm_inad_1)

glm_probs_inad_1 = predict(glm_inad_1, teste_inad_1 ,type = "response")

glm_probs_inad_1[1:10]

contrasts(inad_1$inad1)

dim(teste_inad_1)

glm_pred_inad1 = rep("0" , 36397)
glm_pred_inad1[glm_probs_inad_1 > 0.5] = '1'

teste_inad_1['pred1_logit'] = as.factor(glm_pred_inad1)

head(teste_inad_1)

confusionMatrix(data = teste_inad_1$pred1_logit,  
                reference = teste_inad_1$inad1,
                positive = '1')

glm_inad_2 = glm ( inad2 ~., data = inad_2 , 
                  family = binomial, subset = train2)

dim(teste_inad_2)

glm_probs_inad_2 = predict(glm_inad_2, teste_inad_2 ,type = "response")
glm_pred_inad2 = rep("0" , 34851)
glm_pred_inad2[glm_probs_inad_2 > 0.5] = '1'
teste_inad_2['pred2_logit'] = as.factor(glm_pred_inad2)

confusionMatrix(data = teste_inad_2$pred2_logit,  
                reference = teste_inad_2$inad2,
                positive = '1')

glm_inad_3 = glm ( inad3 ~., data = inad_3 , 
                  family = binomial, subset = train3)

dim(teste_inad_3)

glm_probs_inad_3 = predict(glm_inad_3, teste_inad_3 ,type = "response")
glm_pred_inad3 = rep("0" , 32442)
glm_pred_inad3[glm_probs_inad_3 > 0.5] = '1'
teste_inad_3['pred3_logit'] = as.factor(glm_pred_inad3)

confusionMatrix(data = teste_inad_3$pred3_logit,  
                reference = teste_inad_3$inad3,
                positive = '1')
