using System;
using Microsoft.ML.Data;

namespace StockForecast.Models
{
    public class AtivoPrediction
    {
        [ColumnName("Score")]
        public float ValorPrevisto { get; set; }
    }
}