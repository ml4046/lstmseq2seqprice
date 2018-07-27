# Pattern Extraction and Price Forecasting with LSTM and k-means

Implements a LSTM Composite model in TensorFlow described in [1] by combining an autoencoder and a Seq2Seq for its predictor. Extracted n-day market pattern using k-means from Scikit-Learn. 

## Getting Started
The following are the required packages to run the models. 

### Prerequisites

Python 2.7x<br />
Numpy<br />
Pandas<br />
Tensorflow 1.8.0-rc1<br />
ccxt (optional: retrieve Crypto prices)
### Installing



```
pip install numpy pandas tensorflow
```
In addition if you want to use ccxt to retrieve prices

```
pip install ccxt
```

End with an example of getting some data out of the system or using it for a little demo





## Built With

* [TensorFlow](https://github.com/tensorflow/tensorflow)
* [Scikit-Learn](https://github.com/scikit-learn/scikit-learn)

## License

This project is licensed under the MIT License - see the [LICENSE.md](LICENSE.md) file for details

## References

* [1] https://arxiv.org/pdf/1502.04681.pdf
