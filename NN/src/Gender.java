/**
 * Created by Mahashree on 3/6/2017.
 */

import java.io.*;
import java.util.Random;
import java.util.ArrayList;
import java.util.Collections;
import java.nio.charset.Charset;
import java.nio.MappedByteBuffer;
import java.nio.channels.FileChannel;


public class Gender {

    public static void main(String[] args) {
        ArrayList<Matrix> data = new ArrayList<Matrix>();

        if (args[0].equals("-train")) {
            loadData(data, args[1] + "Male/", 1);
            loadData(data, args[1] + "Female/", 0);
            System.out.println("Training Neural Network...");
            System.out.println("Epoch\tTraining Accuracy");
            train(data);
            System.out.println();
        } else if (args[0].equals("-test")) {
            loadData(data, args[1], -1);
            System.out.println("Testing Neural Network...");
            test(data);
        } else if (args[0].equals("-validate")) {
            loadData(data, args[1] + "Male/", 1);
            loadData(data, args[1] + "Female/", 0);
            crossValidate(data);
        }
    }

    private static void crossValidate(ArrayList<Matrix> data) {

        ArrayList<Double> accuracyListMain = new ArrayList<Double>();
        for (int i = 0; i < 10; i++) {
            Collections.shuffle(data);
            ArrayList<Double> accuracyList = new ArrayList<Double>();
            for(int k=0; k<5; k++) {
                ArrayList<Matrix> randomFold = new ArrayList<Matrix>();
                ArrayList<Matrix> testData = new ArrayList<Matrix>();
                int start = k*(data.size()/5);
                for (int j = start; j < start+(data.size() / 5); j++) {
                    testData.add(data.get(j));
                }
                for (Matrix j : data) {
                    if(!testData.contains(j)) {
                        randomFold.add(j);
                    }
                }
//                System.out.println("testing next fold");
                train(randomFold);
                double score = test(testData);
                accuracyList.add(score);
                accuracyListMain.add(score);
            }
            double overallAccuracy = 0;
            for (double score : accuracyList) {
                overallAccuracy += score;
            }
            System.out.print("mean fold accuracy: ");
            System.out.println(overallAccuracy / accuracyList.size());
            System.out.print("Standard deviation of fold accuracy: ");
            System.out.println(standardDeviation(accuracyList,overallAccuracy / accuracyList.size()));
        }
        double overallAccuracy = 0;
        for (double score : accuracyListMain) {
            overallAccuracy += score;
        }
        System.out.print("Overall accuracy: ");
        System.out.println(overallAccuracy / accuracyListMain.size());

    }


    private static void train(ArrayList<Matrix> data) {
        NeuralNetwork NN = new NeuralNetwork();
        Collections.shuffle(data);
        double error = 0, prev_error = 0;
        for (int i = 0; Math.abs(prev_error - error) > 0.00000005 || i == 0; i++) {
            prev_error = error;
            double correct = 0;
//            for(Matrix j : data) {
//                NN.feed_forward(j);
//            }
//            NN.back_propagate();
//            for(Matrix j : data) {
//
//                int predict = NN.predict() ? 1 : 0;
//                if (predict == j.gender)
//                    correct++;
//            }
            for(Matrix j : data) {
                NN.feed_forward(j);
                NN.back_propagate();
                int predict = NN.predict() ? 1 : 0;
                if (predict == j.gender)
                    correct++;
            }
            error = 1 - (correct / data.size());
            double accuracy = correct / data.size();
//            System.out.println(i + "\t" + accuracy);
        }

        try {
            FileOutputStream fileOut = new FileOutputStream("weights.txt");
            ObjectOutputStream out = new ObjectOutputStream(fileOut);
            out.writeObject(NN);
            out.close();
        } catch (IOException i) {
            i.printStackTrace();
            System.exit(7);
        }
    }


    public static void loadData(ArrayList<Matrix> set, String directory, int gender) {
        for (String image : (new File(directory)).list()) {
            try {
                FileInputStream stream = new FileInputStream(new File(directory + "/" + image));
                FileChannel fc = stream.getChannel();
                MappedByteBuffer buffer = fc.map(FileChannel.MapMode.READ_ONLY, 0, fc.size());
                set.add(new Matrix(directory + "/" + image, Charset.defaultCharset().decode(buffer).toString(), gender));
                stream.close();
            } catch (Exception ex) {
                System.err.println("Error: " + ex.getMessage());
            }
        }
    }

    private static double standardDeviation(ArrayList<Double> confidenceList, double mean){
        double variance = 0;
        for(double x : confidenceList){
            variance+= (x-mean)*(x-mean);
        }
        variance = variance/confidenceList.size();
        return Math.sqrt(variance);
    }

    private static double test(ArrayList<Matrix> data) {
//        System.out.println("\ti\tname\t\tgender\terror\tconfidence");
        NeuralNetwork NN = null;

        double accurary = 0;
        try {
            FileInputStream file_in = new FileInputStream("weights.txt");
            ObjectInputStream in = new ObjectInputStream(file_in);
            NN = (NeuralNetwork)in.readObject();
            in.close();
            file_in.close();
        } catch (IOException i) {
            i.printStackTrace();
            System.exit(5);
        } catch (ClassNotFoundException c) {
            System.out.println("Class not found");
            c.printStackTrace();
            System.exit(6);
        }
        double meanConfidence = 0;
        ArrayList<Double> confidenceList = new ArrayList<Double>();
        for (int i = 0; i < data.size(); i++) {
            NeuralNetwork.OutputNode best = NN.test(data.get(i));
            double confidence = Math.pow(best.target - best.output, 2);
            meanConfidence+=confidence;
            confidenceList.add(confidence);
//            System.out.println("\t"+ i +"\t" + data.get(i).name + "\t" + best.gender + "\t" + NN.predict() + "\t"+ confidence);
            if(best.gender == data.get(i).gender){
                accurary++;
            }
        }
//        System.out.print("Mean confidence: ");
//        System.out.println(meanConfidence/data.size());
//        System.out.print("Standard Deviation of confidence: ");
//        System.out.println(standardDeviation(confidenceList, meanConfidence/data.size()));
        return accurary/data.size();
    }
}

class Matrix {
    public static int length = 128;
    public static int width = 120;
    public int gender;
    public int[] pixels;
    String name;

    public Matrix(String n, String raw, int g) {
        pixels = new int[length * width];
        String[] rawpixels = raw.split("[ \n\r]+");
        gender = g;
        name = n;
        for(int i = 0; i < length * width; i++)
            pixels[i] = Integer.valueOf(rawpixels[i]);
    }
}

class NeuralNetwork implements Serializable {
    int input_layer_size = Matrix.length * Matrix.width;
    int hidden_layer_size = 10;
    int output_layer_size = 2;
    int num_hidden_layers = 1;
    double learning_rate = 0.2;

    InputNode[] input_layer;
    HiddenNode[][] hidden_layers;
    OutputNode[] output_layer;

    class Neuron implements Serializable {
        public double input;
        public double output;
        public double error;
    }

    public class InputNode implements Serializable {
        public double value;
        public double[] weights;
        public InputNode(double[] w) {
            weights = w;
            weights[0] = 1;
        }
    }

    public class HiddenNode extends Neuron implements Serializable {
        public double[] weights;
        public HiddenNode(double[] w) {
            weights = w;
            weights[0] = 1;
        }
    }

    public class OutputNode extends Neuron implements Serializable {
        public double target;
        public int gender;
        public OutputNode(int g) {
            gender = g;
        }
    }

    public NeuralNetwork() {
        Random rand = new Random();
        input_layer = new InputNode[input_layer_size];
        hidden_layers = new HiddenNode[num_hidden_layers][hidden_layer_size];
        output_layer = new OutputNode[output_layer_size];
        output_layer[0] = new OutputNode(1);
        output_layer[1] = new OutputNode(0);

        //randomizing input layer
        for (int i = 0; i < input_layer_size; i++) {
            input_layer[i] = new InputNode(new double[hidden_layer_size]);
            for (int j = 0; j < hidden_layer_size; j++)
                input_layer[i].weights[j] = rand.nextDouble() - 0.5;
        }

        //randomizing hidden layer
        for (int i = 0; i < num_hidden_layers; i++) {
            for (int j = 0; j < hidden_layer_size; j++) {
                if(i != num_hidden_layers - 1) {
                    hidden_layers[i][j] = new HiddenNode(new double[hidden_layer_size]);
                    for(int k = 0; k < hidden_layer_size; k++) {
                        hidden_layers[i][j].weights[k] = rand.nextDouble() - 0.5;
                    }
                } else { //randomizing output layer
                    hidden_layers[i][j] = new HiddenNode(new double[output_layer_size]);
                    for(int k = 0; k < output_layer_size; k++) {
                        hidden_layers[i][j].weights[k] = rand.nextDouble() - 0.5;
                    }
                }
            }
        }
    }

    private double sigmoid(double x) {
        return 1 / (1 + Math.exp(-x));
    }

    public void feed_forward(Matrix face) {
        for(int i = 0; i < input_layer_size; i++)
            input_layer[i].value = face.pixels[i];

        for(int i = 0; i < num_hidden_layers; i++) {
            for(int j = 0; j < hidden_layer_size; j++) {
                double total = 0.0;
                if(i > 0) {
                    for(int k = 0; k < hidden_layer_size; k++)
                        total += hidden_layers[i - 1][k].output * hidden_layers[i - 1][k].weights[j];
                } else {
                    for(int k = 0; k < input_layer_size; k++)
                        total += input_layer[k].value * input_layer[k].weights[j];
                }
                hidden_layers[i][j].input = total;
                hidden_layers[i][j].output = sigmoid(total);
            }
        }

        for(int i = 0; i < output_layer_size; i++) {
            double total = 0;
            for(int j = 0; j < hidden_layer_size; j++)
                total += hidden_layers[num_hidden_layers - 1][j].output * hidden_layers[num_hidden_layers - 1][j].weights[i];

            output_layer[i].input = total;
            output_layer[i].output = sigmoid(total);
            output_layer[i].target = output_layer[i].gender == face.gender ? 1.0 : 0.0;
            output_layer[i].error = (output_layer[i].target - output_layer[i].output) * output_layer[i].output * (1 - output_layer[i].output);
        }
    }

    public void back_propagate() {
        for(int i = num_hidden_layers - 1; i > -1; i--) {
            for(int j = 0; j < hidden_layer_size; j++) {
                double total = 0;
//                if(i != num_hidden_layers - 1) {
//                    for(int k = 0; k < hidden_layer_size; k++) {
//                        total += hidden_layers[i][j].weights[k] * hidden_layers[i + 1][k].error;
//                    }
//                } else {
                    for(int k = 0; k < output_layer_size; k++) {
                        total += hidden_layers[i][j].weights[k] * output_layer[k].error;
                    }
//                }
                hidden_layers[i][j].error = total;
            }
        }
        //input layer weights fixed
        for(int i = 0; i < hidden_layer_size; i++)
            for(int j = 0; j < input_layer_size; j++)
                input_layer[j].weights[i] += learning_rate * hidden_layers[0][i].error * input_layer[j].value;

        //hidden layer weights updated
//        for(int i = 0; i < num_hidden_layers - 1; i++)
//            for(int j = 0; j < hidden_layer_size; j++)
//                for(int k = 0; k < hidden_layer_size; k++)
//                    hidden_layers[i][k].weights[j] += learning_rate * hidden_layers[i + 1][j].error * hidden_layers[i][k].output;

        for(int i = 0; i < output_layer_size; i++)
            for(int j = 0; j < hidden_layer_size; j++)
                hidden_layers[num_hidden_layers - 1][j].weights[i] += learning_rate * output_layer[i].error * hidden_layers[num_hidden_layers - 1][j].output;
    }

    public OutputNode test(Matrix face) {
        //pass in the pictures' values
        for(int i = 0; i < input_layer_size; i++)
            input_layer[i].value = face.pixels[i]; //copy into input layer

        //propogate values through
        for(int i = 0; i < num_hidden_layers; i++) {
            for(int j = 0; j < hidden_layer_size; j++) {
                double total = 0.0;
                if(i > 0) {
                    for(int k = 0; k < hidden_layer_size; k++) {
                        total += hidden_layers[i - 1][k].output * hidden_layers[i - 1][k].weights[j];
                    }
                } else {
                    for(int k = 0; k < input_layer_size; k++) {
                        total += input_layer[k].value * input_layer[k].weights[j];
                    }
                }
                hidden_layers[i][j].input = total;
                hidden_layers[i][j].output = sigmoid(total);
            }
        }

        OutputNode best = new OutputNode(-1);
        best.output = -1;

        for(int i = 0; i < output_layer_size; i++) {
            double total = 0.0;
            for(int j = 0; j < hidden_layer_size; j++)
                total += hidden_layers[num_hidden_layers - 1][j].output * hidden_layers[num_hidden_layers - 1][j].weights[i];
            output_layer[i].input = total;
            output_layer[i].output = sigmoid(total);
            output_layer[i].target = output_layer[i].gender == face.gender ? 1.0 : 0.0;
            output_layer[i].error = (output_layer[i].target - output_layer[i].output) * output_layer[i].output * (1 - output_layer[i].output);

            if(output_layer[i].output > best.output)
                best = output_layer[i];
        }

        return best;
    }

    public boolean predict() {
        return output_layer[0].output > output_layer[1].output;
    }
}

