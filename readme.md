使用cycleGAN进行钢板样本缺陷翻译。  
主文件：
cyclegan_plates.py  
直接运行将进行训练，读取 ./datasets/plates下的数据。  
数据格式如下  
/datasets  
-/plates  
--/trainA  
--/trainB  
--/testA  
--/testB  

训练中模型将被保存在./saved_model文件夹内  
./predict_images文件夹内将保存中途训练可视化结果



待做：
3.  增加一个翻译所有缺陷的工具