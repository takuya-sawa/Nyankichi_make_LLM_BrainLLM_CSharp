using System;
using System.Collections.Generic;
using System.Linq;
using BrainLLM;

namespace BrainLLM;

/// <summary>
/// 大脳システムのデモンストレーション
/// 複数の保存された脳ネットワークを統合して大脳として機能させる
/// </summary>
public class CerebrumDemo
{
    public static void RunDemo()
    {
        Console.WriteLine("\n{'=',-70}");
        Console.WriteLine("  Cerebrum System - 大脳統合システム");
        Console.WriteLine("  複数の脳領域を統合して協調動作");
        Console.WriteLine($"{'=',-70}\n");

        // トークナイザー準備
        var tokenizer = new SimpleTokenizer();
        var allWords = new[] { "hello", "world", "neural", "network", "machine", "learning", "brain", "cells", "models" };
        foreach (var word in allWords)
        {
            tokenizer.AddWord(word);
        }

        int vocabSize = tokenizer.VocabSize;
        const int embeddingDim = 32;
        const int hiddenNeurons = 64;

        // ============================================================
        // ステップ1: 異なる特化型ネットワークを3つ訓練
        // ============================================================
        Console.WriteLine("[Step 1] 3つの特化型脳領域を訓練中...\n");

        // 領域1: 言語処理（全データで訓練）
        Console.WriteLine("  [Region 1] Language Processing - 言語処理領域を訓練");
        var languageNetwork = new BrainNetwork(embeddingDim, hiddenNeurons, vocabSize);
        TrainNetwork(languageNetwork, GetAllTrainingData(), tokenizer, vocabSize, epochs: 15, lr: 0.1f);

        // 領域2: 技術用語特化（neural, machine, learningに特化）
        Console.WriteLine("\n  [Region 2] Technical Processing - 技術用語特化領域を訓練");
        var technicalNetwork = new BrainNetwork(embeddingDim, hiddenNeurons, vocabSize);
        TrainNetwork(technicalNetwork, GetTechnicalData(), tokenizer, vocabSize, epochs: 20, lr: 0.12f);

        // 領域3: 一般会話（hello, brainに特化）
        Console.WriteLine("\n  [Region 3] General Processing - 一般会話領域を訓練");
        var generalNetwork = new BrainNetwork(embeddingDim, hiddenNeurons, vocabSize);
        TrainNetwork(generalNetwork, GetGeneralData(), tokenizer, vocabSize, epochs: 20, lr: 0.12f);

        // ============================================================
        // ステップ2: 大脳の構築と領域の追加
        // ============================================================
        Console.WriteLine("\n{'=',-70}");
        Console.WriteLine("[Step 2] 大脳の構築 - 3つの領域を統合\n");

        var cerebrum = new Cerebrum("Primary Language Cerebrum");
        cerebrum.AddRegion("LanguageArea", languageNetwork, RegionFunction.LanguageProcessing);
        cerebrum.AddRegion("TechnicalArea", technicalNetwork, RegionFunction.PatternRecognition);
        cerebrum.AddRegion("GeneralArea", generalNetwork, RegionFunction.GeneralPurpose);

        cerebrum.PrintStatus();

        // ============================================================
        // ステップ3: 大脳統合推論テスト
        // ============================================================
        Console.WriteLine("{'=',-70}");
        Console.WriteLine("[Step 3] 大脳統合推論テスト\n");

        var testWords = new[] { "hello", "neural", "machine", "brain" };
        var expectedOutputs = new[] { "world", "network", "learning", "cells" };

        for (int i = 0; i < testWords.Length; i++)
        {
            string input = testWords[i];
            string expected = expectedOutputs[i];
            
            Console.WriteLine($"{'─',-70}");
            Console.WriteLine($"入力: '{input}' → 期待: '{expected}'");
            
            var inputVec = OneHotEncode(tokenizer.Encode(input), vocabSize);
            
            // 全領域で並列推論
            var regionOutputs = cerebrum.IntegratedForward(inputVec);

            // 各統合モードでテスト
            Console.WriteLine("\n  [統合モード別の判断]:");
            
            foreach (ConsensusMode mode in Enum.GetValues(typeof(ConsensusMode)))
            {
                var integrated = cerebrum.ConsensusDecision(regionOutputs, mode);
                int predictedId = integrated.Select((val, idx) => (val, idx))
                    .OrderByDescending(x => x.val).First().idx;
                string predicted = tokenizer.Decode(predictedId);
                string result = (predicted == expected) ? "✅" : "❌";
                
                Console.WriteLine($"    {mode,-18}: '{predicted}' {result} (Conf: {integrated[predictedId]:F3})");
            }
            
            Console.WriteLine();
        }

        // ============================================================
        // ステップ4: 大脳の永続化
        // ============================================================
        Console.WriteLine("\n{'=',-70}");
        Console.WriteLine("[Step 4] 大脳の保存と読み込み\n");

        string cerebrumDir = "saved_cerebrum";
        cerebrum.SaveCerebrum(cerebrumDir);

        // 保存した大脳を読み込み
        var loadedCerebrum = Cerebrum.LoadCerebrum(cerebrumDir);
        loadedCerebrum.PrintStatus();

        // 読み込んだ大脳でテスト
        Console.WriteLine("[Loaded Cerebrum Test] 読み込んだ大脳で推論:");
        string testWord = "neural";
        var testVec = OneHotEncode(tokenizer.Encode(testWord), vocabSize);
        var testOutputs = loadedCerebrum.IntegratedForward(testVec);
        var consensus = loadedCerebrum.ConsensusDecision(testOutputs, ConsensusMode.WeightedAverage);
        int predId = consensus.Select((val, idx) => (val, idx)).OrderByDescending(x => x.val).First().idx;
        Console.WriteLine($"  '{testWord}' → '{tokenizer.Decode(predId)}' ✅\n");

        // ============================================================
        // ステップ5: 領域の選択的活性化
        // ============================================================
        Console.WriteLine("{'=',-70}");
        Console.WriteLine("[Step 5] 領域の選択的活性化（技術領域のみ）\n");

        loadedCerebrum.SetRegionActive("GeneralArea", false);
        loadedCerebrum.PrintStatus();

        Console.WriteLine("[Technical Focus Test] 技術領域中心で推論:");
        var activeOutputs = loadedCerebrum.IntegratedForward(testVec);
        var activeCons = loadedCerebrum.ConsensusDecision(activeOutputs, ConsensusMode.WeightedAverage);
        int activeId = activeCons.Select((val, idx) => (val, idx)).OrderByDescending(x => x.val).First().idx;
        Console.WriteLine($"  '{testWord}' → '{tokenizer.Decode(activeId)}' (Active: {activeOutputs.Count} regions)\n");

        // ============================================================
        // ステップ6: 海馬のアクセス経路分析
        // ============================================================
        Console.WriteLine("\n{'=',-70}");
        Console.WriteLine("[Step 6] 海馬のアクセス経路分析\n");

        var hippocampus = loadedCerebrum.GetHippocampus();
        
        Console.WriteLine("[Hippocampus Analysis] 記憶の統合:");
        var consolidated = hippocampus.ConsolidateMemory();
        
        Console.WriteLine("\n[Recent Episodes] 最近のエピソード記憶:");
        var recentEpisodes = hippocampus.GetRecentEpisodes(5);
        foreach (var episode in recentEpisodes)
        {
            Console.WriteLine($"  - t={episode.Timestamp}: {episode.EventName} ({episode.Context})");
        }
        
        Console.WriteLine("\n[Frequent Pathways] 頻繁にアクセスされる経路:");
        var frequentPaths = hippocampus.GetFrequentPathways(minAccessCount: 2);
        foreach (var path in frequentPaths.Take(5))
        {
            Console.WriteLine($"  - Region Access: 強度={path.Strength:F3}, " +
                $"アクセス回数={path.AccessCount}, 初回={path.FirstAccess:HH:mm:ss}");
        }

        // ============================================================
        // ステップ7: 創造的思考テスト（ランダム性）
        // ============================================================
        Console.WriteLine("\n{'=',-70}");
        Console.WriteLine("[Step 7] 創造的思考テスト - ランダム性による新しいアイデア生成\n");

        Console.WriteLine("[Test 1] 探索率を上げて創造的想起:");
        hippocampus.SetExplorationRate(0.8f);  // 80%探索
        
        for (int i = 0; i < 3; i++)
        {
            var recalled = hippocampus.RecallEpisode("Integration");
            if (recalled != null)
            {
                Console.WriteLine($"  試行{i+1}: {recalled.EventName} (Context: {recalled.Context})");
            }
        }

        Console.WriteLine("\n[Test 2] 新しいエピソードの創造:");
        var novelEpisode1 = hippocampus.CreateNovelEpisode("CreativeIdea");
        var novelEpisode2 = hippocampus.CreateNovelEpisode("Innovation");
        
        if (novelEpisode1 != null)
        {
            Console.WriteLine($"  創造1: {novelEpisode1.EventName}");
            Console.WriteLine($"    活性化パターン数: {novelEpisode1.NeuronActivations.Count}");
        }
        
        if (novelEpisode2 != null)
        {
            Console.WriteLine($"  創造2: {novelEpisode2.EventName}");
            Console.WriteLine($"    活性化パターン数: {novelEpisode2.NeuronActivations.Count}");
        }

        Console.WriteLine("\n[Test 3] 探索率を戻して通常モード:");
        hippocampus.SetExplorationRate(0.15f);  // 15%に戻す
        var normalRecall = hippocampus.RecallEpisode("Integration");
        if (normalRecall != null)
        {
            Console.WriteLine($"  通常想起: {normalRecall.EventName}");
        }

        Console.WriteLine("\n{'=',-70}");
        Console.WriteLine("[Done] 大脳システムのデモンストレーション完了！");
        Console.WriteLine("  - 複数の保存されたネットワーク → 統合された大脳");
        Console.WriteLine("  - 領域間の協調動作と統合判断");
        Console.WriteLine("  - 大脳全体の永続化と復元");
        Console.WriteLine("  - 海馬によるアクセス経路の記憶");
        Console.WriteLine("  - ランダム性による創造的思考 ⭐NEW");
        Console.WriteLine($"{'=',-70}\n");
    }

    private static List<(string, string)> GetAllTrainingData()
    {
        return new List<(string, string)>
        {
            ("hello", "world"),
            ("neural", "network"),
            ("machine", "learning"),
            ("brain", "cells"),
            ("learning", "models"),
        };
    }

    private static List<(string, string)> GetTechnicalData()
    {
        return new List<(string, string)>
        {
            ("neural", "network"),
            ("machine", "learning"),
            ("learning", "models"),
        };
    }

    private static List<(string, string)> GetGeneralData()
    {
        return new List<(string, string)>
        {
            ("hello", "world"),
            ("brain", "cells"),
        };
    }

    private static void TrainNetwork(BrainNetwork network, List<(string input, string output)> data, 
        SimpleTokenizer tokenizer, int vocabSize, int epochs, float lr)
    {
        for (int epoch = 0; epoch < epochs; epoch++)
        {
            float totalLoss = 0;
            foreach (var (input, output) in data)
            {
                int inputId = tokenizer.Encode(input);
                int targetId = tokenizer.Encode(output);
                
                var inputVec = OneHotEncode(inputId, vocabSize);
                var outVec = network.Forward(inputVec);
                
                float loss = -(float)Math.Log(Math.Max(outVec[targetId], 1e-10f));
                totalLoss += loss;
                
                network.TrainStep(inputVec, targetId, lr);
            }

            if ((epoch + 1) % 5 == 0)
            {
                Console.WriteLine($"    Epoch {epoch + 1}/{epochs}: Loss = {totalLoss / data.Count:F4}");
            }

            lr *= 0.95f;
        }
    }

    private static float[] OneHotEncode(int id, int vocabSize)
    {
        var vec = new float[vocabSize];
        if (id > 0 && id < vocabSize)
        {
            vec[id] = 1.0f;
        }
        return vec;
    }
}
