using Microsoft.ML;
using Microsoft.ML.Data;
using static Microsoft.ML.DataOperationsCatalog;

namespace MovieRecommendation;

class Program
{
    private static string DatasetFile = "avaliacoes_filmes.csv";
    private static string BasePath = "/MLModel";
    private static string ModelPath = $"{BasePath}/Filmes.zip";

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

        PreverAvaliacao(context, model);

        Console.ReadLine();

    }


    private static TrainTestData SanitizarDados(MLContext context)
    {
        // Lê o arquivo e o transforma em um dataset.
        IDataView dataview = context.Data
            .LoadFromTextFile<AvaliacaoFilme>(DatasetFile, ',', true);

        TrainTestData trainTestData = context.Data.TrainTestSplit(dataview, 0.2);

        return trainTestData;
    }

    private static ITransformer TreinarModelo(MLContext mlContext, IDataView trainData)
    {
        var trainer = mlContext.Recommendation().Trainers.MatrixFactorization(
            labelColumnName: "Label",
            matrixColumnIndexColumnName: "UserIdEncoded",
            matrixRowIndexColumnName: "MovieIdEncoded",
            numberOfIterations: 20,
            approximationRank: 100
            );

        var pipeline = mlContext.Transforms.Conversion.ConvertType(outputColumnName: "Label", inputColumnName: "Avaliacao", outputKind: DataKind.Single)
               .Append(mlContext.Transforms.Conversion.MapValueToKey(outputColumnName: "UserIdEncoded", inputColumnName: "IDUsuario"))
               .Append(mlContext.Transforms.Conversion.MapValueToKey(outputColumnName: "MovieIdEncoded", inputColumnName: "IDFilme"))
               .Append(trainer);

        ITransformer model = pipeline.Fit(trainData);

        return model;
    }

    private static RegressionMetrics AvaliarDesempenhoModelo(MLContext context, ITransformer model,
        IDataView testSet)
    {
        IDataView predictions = model.Transform(testSet);

        RegressionMetrics metrics = context.Regression.Evaluate(predictions);

        return metrics;
    }

    private static void SalvarModelo(MLContext context, ITransformer model,
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

        context.Model.Save(model, schema, ModelPath);
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

    private static void PreverAvaliacao(MLContext mlContext, ITransformer model)
    {
        var predictionEngine = mlContext.Model.CreatePredictionEngine<AvaliacaoFilme, PrevisaoAvaliacaoFilme>(model);

        var prediction = predictionEngine.Predict( new AvaliacaoFilme { IDUsuario = 1, IDFilme = 103 } );
        Console.WriteLine($"O usuário 1 avaliaria o filme 103 com uma nota de {prediction.Score}.");

        prediction = predictionEngine.Predict(new AvaliacaoFilme { IDUsuario = 2, IDFilme = 103 });
        Console.WriteLine($"O usuário 2 avaliaria o filme 103 com uma nota de {prediction.Score}.");

        prediction = predictionEngine.Predict(new AvaliacaoFilme { IDUsuario = 3, IDFilme = 103 });
        Console.WriteLine($"O usuário 3 avaliaria o filme 103 com uma nota de {prediction.Score}.");

        prediction = predictionEngine.Predict(new AvaliacaoFilme { IDUsuario = 4, IDFilme = 103 });
        Console.WriteLine($"O usuário 4 avaliaria o filme 103 com uma nota de {prediction.Score}.");

        prediction = predictionEngine.Predict(new AvaliacaoFilme { IDUsuario = 5, IDFilme = 103 });
        Console.WriteLine($"O usuário 5 avaliaria o filme 103 com uma nota de {prediction.Score}.");
    }

}

public class AvaliacaoFilme
{
    [LoadColumn(0)]
    public float IDUsuario { get; set; }

    [LoadColumn(1)]
    public float IDFilme { get; set; }

    [LoadColumn(2)]
    public string NomeFilme { get; set; }

    [LoadColumn(3)]
    public float Avaliacao { get; set; }
}

public class PrevisaoAvaliacaoFilme
{
    public float Label { get; set; }
    public float Score { get; set; }
}


