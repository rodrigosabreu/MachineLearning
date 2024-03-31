using Microsoft.ML;
using Microsoft.ML.Data;
using StockForecast.Models;
using static Microsoft.ML.DataOperationsCatalog;

namespace StockForecast
{
    class Program
    {
        private static string DatasetFile = "PETR4.SA.csv";
        private static string BasePath = "/MLModel";
        private static string ModelPath = $"{BasePath}/StockForecast.zip";

        static void Main(string[] args)
        {
            // Cria o contexto que trabalhará com aprendizado de máquina.
            MLContext context = new MLContext();

            // Lê o arquivo e o transforma em um dataset.
            TrainTestData splitData = SanitizarDados(context);

            ITransformer model = TreinarModelo(context, splitData.TrainSet);

            RegressionMetrics metrics = AvaliarDesempenhoModelo(context, model, splitData.TestSet);

            SalvarModelo(context, model, splitData.TrainSet.Schema);

            ImprimirMetricas(metrics);

            PreverPrecosDasAcoes(context, model);

            Console.ReadLine();
        }

        private static TrainTestData SanitizarDados(MLContext context)
        {
            // Lê o arquivo e o transforma em um dataset.
            IDataView dataview = context.Data
            .LoadFromTextFile<Ativo>(DatasetFile, ',', true);

            // Remove as linhas que contiverem algum valor nulo.
            dataview = context.Data.FilterRowsByMissingValues(dataview, "Open",
            "High", "Low", "AdjustedClose", "Volume");

            // Divide o dataset em uma base de treino (80%) e uma de teste (20%).
            TrainTestData trainTestData = context.Data.TrainTestSplit(dataview, 0.2);

            return trainTestData;
        }

        private static ITransformer TreinarModelo(MLContext mlContext, IDataView trainData)
        {
            var treinador = mlContext.Regression.Trainers.Sdca();

            string[] colunas = { "Open", "High", "Low", "Volume" };

            // Constroi o fluxo de transformação de dados e processamento do modelo.
            IEstimator<ITransformer> pipeline = mlContext.Transforms
            .CopyColumns("Label", "Close")
            .Append(mlContext.Transforms.Concatenate("Features", colunas))
            .Append(mlContext.Transforms.NormalizeMinMax("Features"))
            .AppendCacheCheckpoint(mlContext)
            .Append(treinador);

            ITransformer model = pipeline.Fit(trainData);

            return model;
        }

        private static RegressionMetrics AvaliarDesempenhoModelo(MLContext mlContext, ITransformer model,
        IDataView testSet)
        {
            IDataView predictions = model.Transform(testSet);

            RegressionMetrics metrics = mlContext.Regression.Evaluate(predictions);

            return metrics;
        }

        private static void SalvarModelo(MLContext mlContext, ITransformer model,
        DataViewSchema schema)
        {
            if (!Directory.Exists(BasePath))
            {
                Directory.CreateDirectory(BasePath);
            }
            else
            {
                foreach (String file in Directory.EnumerateFiles(BasePath))
                {
                    File.Delete(file);
                }
            }

            mlContext.Model.Save(model, schema, ModelPath);
        }

        private static void ImprimirMetricas(RegressionMetrics metrics)
        {
            Console.WriteLine("-------------------- MÉTRICAS --------------------");
            Console.WriteLine($"Mean Absolute Error: {metrics.MeanAbsoluteError}");
            Console.WriteLine($"Mean Squared Error: {metrics.MeanSquaredError}");
            Console.WriteLine($"Root Mean Squared Error: {metrics.RootMeanSquaredError}");
            Console.WriteLine($"R Squared: {metrics.RSquared}");
            Console.WriteLine("--------------------------------------------------");
        }

        private static void PreverPrecosDasAcoes(MLContext context, ITransformer model)
        {
            Ativo[] stocks = {
                new Ativo
                {
                    Open = 25.700001f,
                    High = 25.780001f,
                    Low = 25.430000f,
                    Close = 25.450001f,
                    AdjustedClose = 21.730824f,
                    Volume = 17841800
                },
                new Ativo
                {
                    Open = 30.799999f,
                    High = 30.889999f,
                    Low = 29.750000f,
                    Close = 29.920000f,
                    AdjustedClose = 29.918381f,
                    Volume = 73522900
                },
                new Ativo
                {
                    Open = 16.670000f,
                    High = 16.760000f,
                    Low = 15.530000f,
                    Close = 15.720000f,
                    AdjustedClose = 15.719150f,
                    Volume = 115633300
                },
                new Ativo
                {
                    Open = 17.51f,
                    High = 17.62f,
                    Low = 17.20f,
                    Close = 17.25f,
                    AdjustedClose = 0f,
                    Volume = 0
                }
            };

            PredictionEngine<Ativo, AtivoPrediction> predictor = context.Model
            .CreatePredictionEngine<Ativo, AtivoPrediction>(model);

            foreach (Ativo stock in stocks)
            {
                AtivoPrediction prediction = predictor.Predict(stock);

                Console.WriteLine("---------------- PREVISÃO ----------------");
                Console.WriteLine($"O preço previsto para a ação é de R$ {prediction.ValorPrevisto:0.#0}");
                Console.WriteLine($"O preço atual é de R$ {stock.Close:0.#0}");
                Console.WriteLine($"Diferença de R$ {prediction.ValorPrevisto - stock.Close:0.#0}");
                Console.WriteLine("------------------------------------------");
            }
        }
    }
}
