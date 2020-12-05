# 5001 Kaggle competition 


```bash
docker build -t notebook .
docker run --runtime=nvidia --rm -dit -v $PWD:/py -w /py -p 20080:2233 notebook bash
```

`lgb3_byday.py` - main script for submission model. (plotting, training, model dump, prediction)

`lgb_para_opt.py` - hyper params tuning lib

`download_wwo.py` - download weather data via API

`requirments.txt` - library list

`*.pkl` - models dumped for prediction, the number repesent the MSE , but larger MSE better performance on leaderboard ,seems `overfitted`,