using System;
using System.Collections.Generic;
using System.Linq;
using BrainLLM;

// ============================================================
// 脳細胞型LLM - BrainLLM
// ニューロンベースの言語モデル
// ============================================================

// コマンドライン引数で大脳デモを起動
if (args.Length > 0 && args[0] == "--cerebrum")
{
    CerebrumDemo.RunDemo();
    return;
}

Console.WriteLine("================================================");
Console.WriteLine("  BrainLLM - Biologically-Inspired Neural Network");
Console.WriteLine("  Brain-cell type neuron-based Language Model");
Console.WriteLine("================================================");
Console.WriteLine("  オプション: --cerebrum で大脳統合システムを実行");
Console.WriteLine("================================================\n");

// ============================================================
// ステップ1：トレーニングデータの準備
// ============================================================
Console.WriteLine("[Data] Training text:");
// Training data: ("input word" -> "output word")
var trainingWords = new List<(string input, string output)>
{
    ("hello", "world"),
    ("neural", "network"),
    ("machine", "learning"),
    ("brain", "cells"),
    ("learning", "models"),
};

foreach (var (input, output) in trainingWords)
{
    Console.WriteLine($"  '{input}' -> '{output}'");
}
Console.WriteLine();

// ============================================================
// ステップ2：トークナイザーの初期化
// ============================================================
var tokenizer = new SimpleTokenizer();
// すべての単語（入力とターゲット）を語彙に追加
var allWords = new[] { "hello", "world", "neural", "network", "machine", "learning", "brain", "cells", "models" };
foreach (var word in allWords)
{
    tokenizer.AddWord(word);
}
Console.WriteLine($"[Tokenizer] Vocab Size: {tokenizer.VocabSize}\n");

// ============================================================
// Step 3: Brain Network Construction
// ============================================================
const int embeddingDim = 32;
const int hiddenNeurons = 64;
int vocabSize = tokenizer.VocabSize;

var brainNetwork = new BrainNetwork(embeddingDim, hiddenNeurons, vocabSize);
brainNetwork.PrintStats();

Console.WriteLine("[Memory] State Memory: Each neuron stores complete state snapshots");
Console.WriteLine("  - Neuron state (weights, action potential) managed as Queue history");
Console.WriteLine("  - Past state restoration and replay enabled (Synaptic Replay)");
Console.WriteLine("[Saltatory] Saltatory Conduction: Direct hidden->output layer connections (learnable)\n");

// ============================================================
// Step 4: Training
// ============================================================
Console.WriteLine("================================================");
Console.WriteLine("  Training Phase");
Console.WriteLine("================================================\n");

const int epochs = 20;
float learningRate = 0.1f;

for (int epoch = 0; epoch < epochs; epoch++)
{
    float totalLoss = 0;
    int steps = 0;

    foreach (var (inputWord, outputWord) in trainingWords)
    {
        // テキストをベクトルに変換
        var input = new float[embeddingDim];
        int tokenId = tokenizer.Encode(inputWord);
        int targetWordId = tokenizer.Encode(outputWord);
        
        if (tokenId > 0 && tokenId < embeddingDim)
        {
            input[tokenId] = 1.0f;  // One-hot encoding
        }

        // Forward pass
        var output = brainNetwork.Forward(input);

        // デバッグ: 最初のエポック、最初のステップのみ
        if (epoch == 0 && steps == 0)
        {
            Console.WriteLine($"[DEBUG Epoch 1, Step 1]");
            Console.WriteLine($"  Input: '{inputWord}' (ID:{tokenId}) → Target: '{outputWord}' (ID:{targetWordId})");
            Console.WriteLine($"  Output probs: {string.Join(", ", output.Select(x => $"{x:F4}"))}");
        }

        // 損失計算
        float loss = -(float)Math.Log(Math.Max(output[targetWordId], 1e-10f));
        totalLoss += loss;

        // バックプロップ
        brainNetwork.TrainStep(input, targetWordId, learningRate);

        steps++;
    }

    float avgLoss = totalLoss / steps;
    Console.WriteLine($"[Epoch {epoch + 1}/{epochs}]");
    Console.WriteLine($"  Average Loss: {avgLoss:F6}");
    Console.WriteLine($"  Learning Rate: {learningRate:F6}\n");

    // Learning rate decay
    learningRate *= 0.9f;
}

// One-Hot Encoding function
static List<float> OneHotEncode(int id, int vocabSize)
{
    var vec = new List<float>(new float[vocabSize]);
    if (id > 0 && id < vocabSize)
    {
        vec[id] = 1.0f;
    }
    return vec;
}

// ============================================================
// Step 5: Inference Test
// ============================================================
Console.WriteLine("================================================");
Console.WriteLine("  Inference Phase");
Console.WriteLine("================================================\n");

var testPairs = new (string input, string expected)[]
{
    ("hello", "world"),
    ("neural", "network"),
    ("machine", "learning"),
    ("brain", "cells"),
    ("learning", "models")
};

foreach (var (word, expected) in testPairs)
{
    var input = new float[embeddingDim];
    int tokenId = tokenizer.Encode(word);
    if (tokenId > 0 && tokenId < embeddingDim)
    {
        input[tokenId] = 1.0f;
    }

    var output = brainNetwork.Forward(input);
    int predictedId = Array.IndexOf(output, output.Max());
    string predictedWord = tokenizer.Decode(predictedId);
    
    string result = (predictedWord == expected) ? "[OK]" : "[NG]";

    Console.WriteLine($"Input: '{word}' -> Predicted: '{predictedWord}' (Expected: '{expected}') {result} (Confidence: {output[predictedId]:F3})");
}

// ============================================================
// Step 6: Brain Network Persistence Test
// ============================================================
Console.WriteLine("\n================================================");
Console.WriteLine("  Brain Persistence Test");
Console.WriteLine("================================================\n");

string brainSavePath = "trained_brain.json";

// Save trained network
brainNetwork.SaveBrain(brainSavePath);
Console.WriteLine($"[Save] Brain saved to: {brainSavePath}");

// Load saved network
var loadedBrain = BrainNetwork.LoadBrain(brainSavePath);
Console.WriteLine($"[Load] Brain loaded from: {brainSavePath}");

// Test inference with loaded network
Console.WriteLine("\n[Loaded Brain Test] Inference with loaded brain:");
foreach (var (input, expected) in trainingWords.Take(3))
{
    var inputVec = OneHotEncode(tokenizer.Encode(input), vocabSize);
    var output = loadedBrain.Forward(inputVec.ToArray());
    int predictedId = output.Select((val, idx) => (val, idx)).OrderByDescending(x => x.val).First().idx;
    string predictedWord = tokenizer.Decode(predictedId);
    string result = (predictedWord == expected) ? "[OK]" : "[NG]";
    Console.WriteLine($"  '{input}' -> '{predictedWord}' {result}");
}

Console.WriteLine("\n[Done] Brain-cell LLM training and inference completed!");
Console.WriteLine("[Done] Trained network persisted to file!\n");

// ============================================================
// Simple Tokenizer
// ============================================================
public class SimpleTokenizer
{
    private Dictionary<string, int> word2id = new();
    private Dictionary<int, string> id2word = new();
    private int nextId = 1;

    public int VocabSize => word2id.Count + 1;

    public void AddWord(string word)
    {
        if (!word2id.ContainsKey(word))
        {
            word2id[word] = nextId;
            id2word[nextId] = word;
            nextId++;
        }
    }

    public int Encode(string word)
    {
        return word2id.TryGetValue(word, out var id) ? id : 0;  // 0 = <unk>
    }

    public string Decode(int id)
    {
        return id2word.TryGetValue(id, out var word) ? word : "<unk>";
    }
}
