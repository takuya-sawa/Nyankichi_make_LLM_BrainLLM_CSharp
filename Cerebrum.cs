using System;
using System.Collections.Generic;
using System.Linq;
using System.Text.Json;
using System.Text.Json.Serialization;
using System.IO;

namespace BrainLLM;

/// <summary>
/// 大脳 - 複数の脳領域（BrainNetwork）を統合する上位システム
/// 生物学的には: 大脳皮質の異なる領域が協調して動作するイメージ
/// </summary>
public class Cerebrum
{
    private Dictionary<string, BrainRegion> regions = new();
    private string cerebrumName;
    private Hippocampus hippocampus;  // 海馬：アクセス経路の記憶
    
    public Cerebrum(string name = "Primary Cerebrum")
    {
        cerebrumName = name;
        hippocampus = new Hippocampus();
    }
    
    /// <summary>
    /// 海馬を取得
    /// </summary>
    public Hippocampus GetHippocampus() => hippocampus;

    /// <summary>
    /// 脳領域を追加（前頭葉、側頭葉、頭頂葉など）
    /// </summary>
    public void AddRegion(string regionName, BrainNetwork network, RegionFunction function)
    {
        regions[regionName] = new BrainRegion
        {
            Name = regionName,
            Network = network,
            Function = function,
            IsActive = true,
            CreatedAt = DateTime.UtcNow
        };
        
        Console.WriteLine($"[Cerebrum] 脳領域を追加: {regionName} ({function})");
    }

    /// <summary>
    /// 保存された脳ネットワークから領域を読み込み
    /// </summary>
    public void LoadRegionFromFile(string regionName, string filePath, RegionFunction function)
    {
        var network = BrainNetwork.LoadBrain(filePath);
        AddRegion(regionName, network, function);
    }

    /// <summary>
    /// 領域を保存
    /// </summary>
    public void SaveRegion(string regionName, string filePath)
    {
        if (regions.TryGetValue(regionName, out var region))
        {
            region.Network.SaveBrain(filePath);
            Console.WriteLine($"[Cerebrum] 領域を保存: {regionName} → {filePath}");
        }
    }

    /// <summary>
    /// 複数領域を統合して推論（大脳統合処理）
    /// </summary>
    public Dictionary<string, float[]> IntegratedForward(float[] input, List<string> activeRegionNames = null)
    {
        var results = new Dictionary<string, float[]>();
        var targetRegions = activeRegionNames ?? regions.Keys.ToList();

        Console.WriteLine($"\n[Cerebrum Integration] {targetRegions.Count}個の脳領域で並列処理:");
        
        foreach (var regionName in targetRegions)
        {
            if (regions.TryGetValue(regionName, out var region) && region.IsActive)
            {
                var output = region.Network.Forward(input);
                results[regionName] = output;
                
                var maxIdx = output.Select((val, idx) => (val, idx))
                    .OrderByDescending(x => x.val).First().idx;
                Console.WriteLine($"  - {regionName}: 最大活性 Index={maxIdx}, Conf={output[maxIdx]:F3}");
                
                // 海馬にアクセス記録
                hippocampus.RecordRegionAccess("Input", regionName, output);
            }
        }
        
        // エピソード記憶として保存
        var activationMap = results.SelectMany(r => r.Value.Select((v, i) => (i, v)))
            .GroupBy(x => x.i)
            .ToDictionary(g => g.Key, g => g.Average(x => x.v));
        hippocampus.SaveEpisode($"Integration_{targetRegions.Count}regions", activationMap, string.Join(",", targetRegions));

        return results;
    }

    /// <summary>
    /// 大脳レベルの統合判断（多数決・重み付け平均など）
    /// </summary>
    public float[] ConsensusDecision(Dictionary<string, float[]> regionOutputs, ConsensusMode mode = ConsensusMode.WeightedAverage)
    {
        if (regionOutputs.Count == 0)
            throw new InvalidOperationException("No region outputs to integrate");

        int outputSize = regionOutputs.First().Value.Length;
        var integrated = new float[outputSize];

        switch (mode)
        {
            case ConsensusMode.WeightedAverage:
                // 重み付け平均（全領域の出力を平均）
                foreach (var output in regionOutputs.Values)
                {
                    for (int i = 0; i < outputSize; i++)
                    {
                        integrated[i] += output[i];
                    }
                }
                
                for (int i = 0; i < outputSize; i++)
                {
                    integrated[i] /= regionOutputs.Count;
                }
                break;

            case ConsensusMode.Voting:
                // 投票方式（最も多くの領域が支持する選択肢）
                var votes = new int[outputSize];
                foreach (var output in regionOutputs.Values)
                {
                    int maxIdx = output.Select((val, idx) => (val, idx))
                        .OrderByDescending(x => x.val).First().idx;
                    votes[maxIdx]++;
                }
                
                int winnerIdx = votes.Select((count, idx) => (count, idx))
                    .OrderByDescending(x => x.count).First().idx;
                integrated[winnerIdx] = 1.0f;
                break;

            case ConsensusMode.MaxPooling:
                // 最大値プーリング（各インデックスで最大値を取る）
                foreach (var output in regionOutputs.Values)
                {
                    for (int i = 0; i < outputSize; i++)
                    {
                        integrated[i] = Math.Max(integrated[i], output[i]);
                    }
                }
                break;
        }

        return integrated;
    }

    /// <summary>
    /// 大脳全体を保存（全領域のメタデータ + 各領域への参照）
    /// </summary>
    public void SaveCerebrum(string directoryPath)
    {
        if (!Directory.Exists(directoryPath))
        {
            Directory.CreateDirectory(directoryPath);
        }

        var cerebrumData = new CerebrumData
        {
            Name = cerebrumName,
            Timestamp = DateTime.UtcNow.Ticks,
            Regions = new List<RegionMetadata>()
        };

        // 各領域を個別ファイルに保存
        foreach (var (regionName, region) in regions)
        {
            string regionFile = Path.Combine(directoryPath, $"{regionName}.json");
            region.Network.SaveBrain(regionFile);

            cerebrumData.Regions.Add(new RegionMetadata
            {
                Name = regionName,
                Function = region.Function,
                FilePath = regionFile,
                IsActive = region.IsActive,
                CreatedAt = region.CreatedAt.Ticks
            });
        }

        // メタデータを保存
        string metaFile = Path.Combine(directoryPath, "cerebrum_meta.json");
        var options = new JsonSerializerOptions { WriteIndented = true };
        string json = JsonSerializer.Serialize(cerebrumData, options);
        File.WriteAllText(metaFile, json);
        
        // 海馬を保存
        string hippocampusFile = Path.Combine(directoryPath, "hippocampus.json");
        hippocampus.Save(hippocampusFile);

        Console.WriteLine($"\n[Cerebrum] 大脳全体を保存しました: {directoryPath}");
        Console.WriteLine($"[Cerebrum] 領域数: {regions.Count}");
        Console.WriteLine($"[Cerebrum] 海馬も保存されました");
    }

    /// <summary>
    /// 大脳全体を読み込み（メタデータから各領域を復元）
    /// </summary>
    public static Cerebrum LoadCerebrum(string directoryPath)
    {
        string metaFile = Path.Combine(directoryPath, "cerebrum_meta.json");
        if (!File.Exists(metaFile))
        {
            throw new FileNotFoundException($"Cerebrum metadata not found: {metaFile}");
        }

        string json = File.ReadAllText(metaFile);
        var cerebrumData = JsonSerializer.Deserialize<CerebrumData>(json)
            ?? throw new InvalidOperationException("Failed to deserialize cerebrum data");

        var cerebrum = new Cerebrum(cerebrumData.Name);

        // 各領域を読み込み
        foreach (var regionMeta in cerebrumData.Regions)
        {
            if (File.Exists(regionMeta.FilePath))
            {
                cerebrum.LoadRegionFromFile(regionMeta.Name, regionMeta.FilePath, regionMeta.Function);
                cerebrum.regions[regionMeta.Name].IsActive = regionMeta.IsActive;
            }
            else
            {
                Console.WriteLine($"[Warning] Region file not found: {regionMeta.FilePath}");
            }
        }

        // 海馬を読み込み
        string hippocampusFile = Path.Combine(directoryPath, "hippocampus.json");
        if (File.Exists(hippocampusFile))
        {
            cerebrum.hippocampus = Hippocampus.Load(hippocampusFile);
        }
        else
        {
            Console.WriteLine("[Warning] Hippocampus file not found, using new instance");
        }
        
        Console.WriteLine($"\n[Cerebrum] 大脳を読み込みました: {directoryPath}");
        Console.WriteLine($"[Cerebrum] 領域数: {cerebrum.regions.Count}");

        return cerebrum;
    }

    /// <summary>
    /// 大脳の状態を表示
    /// </summary>
    public void PrintStatus()
    {
        Console.WriteLine($"\n{'=',-60}");
        Console.WriteLine($"  Cerebrum Status: {cerebrumName}");
        Console.WriteLine($"{'=',-60}");
        Console.WriteLine($"Total Regions: {regions.Count}");
        Console.WriteLine($"Active Regions: {regions.Count(r => r.Value.IsActive)}");
        Console.WriteLine();

        foreach (var (name, region) in regions.OrderBy(r => r.Key))
        {
            string status = region.IsActive ? "🟢 Active" : "⚪ Inactive";
            Console.WriteLine($"  [{status}] {name,-20} ({region.Function})");
        }
        
        // 海馬の統計
        hippocampus.PrintStats();
        Console.WriteLine();
    }

    /// <summary>
    /// 領域を有効化/無効化
    /// </summary>
    public void SetRegionActive(string regionName, bool isActive)
    {
        if (regions.TryGetValue(regionName, out var region))
        {
            region.IsActive = isActive;
            string status = isActive ? "有効化" : "無効化";
            Console.WriteLine($"[Cerebrum] {regionName} を{status}しました");
        }
    }
}

/// <summary>
/// 脳領域（前頭葉、側頭葉など）
/// </summary>
public class BrainRegion
{
    public string Name { get; set; } = "";
    public BrainNetwork Network { get; set; } = null!;
    public RegionFunction Function { get; set; }
    public bool IsActive { get; set; }
    public DateTime CreatedAt { get; set; }
}

/// <summary>
/// 脳領域の機能分類
/// </summary>
public enum RegionFunction
{
    LanguageProcessing,      // 言語処理（ブローカ野など）
    MemoryRetrieval,         // 記憶検索（海馬など）
    DecisionMaking,          // 意思決定（前頭前野など）
    PatternRecognition,      // パターン認識（後頭葉など）
    AttentionControl,        // 注意制御
    EmotionalProcessing,     // 感情処理（扁桃体など）
    MotorControl,            // 運動制御
    GeneralPurpose           // 汎用
}

/// <summary>
/// 統合判断モード
/// </summary>
public enum ConsensusMode
{
    WeightedAverage,  // 重み付け平均
    Voting,           // 投票
    MaxPooling        // 最大値プーリング
}

/// <summary>
/// 大脳全体のシリアライゼーションデータ
/// </summary>
public class CerebrumData
{
    public string Name { get; set; } = "";
    public long Timestamp { get; set; }
    public List<RegionMetadata> Regions { get; set; } = new();
}

/// <summary>
/// 領域メタデータ
/// </summary>
public class RegionMetadata
{
    public string Name { get; set; } = "";
    public RegionFunction Function { get; set; }
    public string FilePath { get; set; } = "";
    public bool IsActive { get; set; }
    public long CreatedAt { get; set; }
}
