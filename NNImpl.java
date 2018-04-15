import java.util.*;

/**
 * The main class that handles the entire network
 * Has multiple attributes each with its own use
 */

public class NNImpl {
    private ArrayList<Node> inputNodes; //list of the output layer nodes.
    private ArrayList<Node> hiddenNodes;    //list of the hidden layer nodes
    private ArrayList<Node> outputNodes;    // list of the output layer nodes

    private ArrayList<Instance> trainingSet;    //the training set

    private double learningRate;    // variable to store the learning rate
    private int maxEpoch;   // variable to store the maximum number of epochs
    private Random random;  // random number generator to shuffle the training set

    /**
     * This constructor creates the nodes necessary for the neural network
     * Also connects the nodes of different layers
     * After calling the constructor the last node of both inputNodes and
     * hiddenNodes will be bias nodes.
     */

    NNImpl(ArrayList<Instance> trainingSet, int hiddenNodeCount, Double learningRate, int maxEpoch, Random random, Double[][] hiddenWeights, Double[][] outputWeights) {
        this.trainingSet = trainingSet;
        this.learningRate = learningRate;
        this.maxEpoch = maxEpoch;
        this.random = random;

        //input layer nodes
        inputNodes = new ArrayList<>();
        int inputNodeCount = trainingSet.get(0).attributes.size();
        int outputNodeCount = trainingSet.get(0).classValues.size();
        for (int i = 0; i < inputNodeCount; i++) {
            Node node = new Node(0);
            inputNodes.add(node);
        }

        //bias node from input layer to hidden
        Node biasToHidden = new Node(1);
        inputNodes.add(biasToHidden);

        //hidden layer nodes
        hiddenNodes = new ArrayList<>();
        for (int i = 0; i < hiddenNodeCount; i++) {
            Node node = new Node(2);
            //Connecting hidden layer nodes with input layer nodes
            for (int j = 0; j < inputNodes.size(); j++) {
                NodeWeightPair nwp = new NodeWeightPair(inputNodes.get(j), hiddenWeights[i][j]);
                node.parents.add(nwp);
            }
            hiddenNodes.add(node);
        }

        //bias node from hidden layer to output
        Node biasToOutput = new Node(3);
        hiddenNodes.add(biasToOutput);

        //Output node layer
        outputNodes = new ArrayList<>();
        for (int i = 0; i < outputNodeCount; i++) {
            Node node = new Node(4);
            //Connecting output layer nodes with hidden layer nodes
            for (int j = 0; j < hiddenNodes.size(); j++) {
                NodeWeightPair nwp = new NodeWeightPair(hiddenNodes.get(j), outputWeights[i][j]);
                node.parents.add(nwp);
            }
            outputNodes.add(node);
        }
    }

    /**
     * Get the prediction from the neural network for a single instance
     * Return the idx with highest output values. For example if the outputs
     * of the outputNodes are [0.1, 0.5, 0.2], it should return 1.
     * The parameter is a single instance
     */

    public int predict(Instance instance) { // ultimately, we want output
        // TODO: add code here
        // we are given one instance
        // instance contains an array of attributes
        // an attribute contains an array of input values -> so they are # of input?

       // System.out.println("Size of the attributes" +  instance.attributes.size());
        for (int i = 0; i < instance.attributes.size(); i++) { // get the size of an instance
            // set the input values for input nodes
            inputNodes.get(i).setInput(instance.attributes.get(i));
            //System.out.println("instance of input: " + instance.attributes.get(i));
        }

      // for (Node in : inputNodes ) {
       //     System.out.println("output of input nodes:" + in.getOutput());
        //}

        // we need to pass in total outputSums to calculateOutput function

        // calculate output for hidden nodes

       //  System.out.println("Size of the hidden nodes" +  hiddenNodes.size());
        for (Node hid: hiddenNodes ) {
            hid.calculateOutput(weightedSum());
            
        }

        // test to see output of hidden nodes
        // for (Node testHid : hiddenNodes ) {
        //     System.out.println("output of hiddenNodes nodes:" + testHid.getOutput());
        // }


       //  System.out.println("Size of the outputNodes nodes" +  outputNodes.size());
        // calculate output for outputNodes


         //double temp = 0.0;
        for (Node out : outputNodes ) {
            out.calculateOutput(weightedSum());  
           // System.out.println("output of output nodes:" + out.getOutput());
            //temp += out.getOutput();
        }

        //System.out.println(temp);

        // given input for input nodes, we can calculate values of hidden nodes
       // for (int j = 0;j < hiddenNodes.size();j++) {
        //    hiddenNodes.get(j).getOutput();
       // }

       // for (int k = 0; k < outputNodes.size();k++) {
       //     outputNodes.get(k).getOutput();; // get nodes and calculate output 
       // }

        int ret = 0;
        double temp = outputNodes.get(0).getOutput();
      //  System.out.println("output of first: " + temp);

        for (int j = 1; j < outputNodes.size(); j++) {
            if (temp < outputNodes.get(j).getOutput()) {
                ret = j;
            }
        }

       // System.out.println(ret);

        return ret;
    }

    /**
     * Train the neural networks with the given parameters
     * <p>
     * The parameters are stored as attributes of this class
     */

    //trains the neural network using a training set, fixed learning rate, 
    //and number of epochs (provided as input to the program).

    public void train() {

        
        // TODO: add code here
        // list all the stuff that I need in here:
        // Each pass through all of the training examples is called an epoch
        for (int i = 0; i < maxEpoch; i++) { // for the number of epoch
            Collections.shuffle(trainingSet,random);
            for (int j = 0; j < trainingSet.size(); j++) { // for the number of instances in a training set
                    // now we are in an instance of a training set
                    // we alet's say we want to training given an instance
                    // what is the first step we have to take to train in an instance

                    int outPut = predict(trainingSet.get(j)); // calculate outPut for output nodes
                    // update gradient

                    // get teacher output
                    int teacher = 0;

                    for (int k = 0; k < trainingSet.get(k).classValues.size();k++) {
                     //    System.out.println("classValues: " + trainingSet.get(0).classValues.get(k));
                        if (trainingSet.get(j).classValues.get(k) == 1)
                         teacher = k;
                    }

                    // System.out.println("teacher get from classValues" + teacher);

                    for (int m = 0;m < outputNodes.size();m++ ) {
                        //trainingSet.get(0).classValues.get(m);
                        if (teacher == m) {
                            outputNodes.get(m).calculateDelta(1);
                       //     System.out.println("delta values: " + outputNodes.get(m).getDelta());
                        }
                        else {
                            outputNodes.get(m).calculateDelta(0);
                           // System.out.println("delta values: " + outputNodes.get(m).getDelta());
                        }
                    }

                    // goal: calculate delta for hidden nodes
                    // we need to know g'(z) * wjk * 


                    // probably should go from the output back to the hidden nodes
                    // easier to do things



                    for (int n = 0; n < hiddenNodes.size(); n++ ) {
                        
                        double gradient = 0.0;
                        // for (Node out : outputNodes ) {
                        //   gradient = out.delta * out.parents.
                        // }

                        // iterative thru outputNodes
                        for (int b = 0; b < outputNodes.size(); b++) {

                           gradient += (outputNodes.get(b).getDelta() * outputNodes.get(b).parents.get(n).weight);
                        }

                        hiddenNodes.get(n).setGradient(gradient);

                       // we are doing 
                    }

                    //given outputGradient, we can calculate delta of hidden notes
                    for (Node hid : hiddenNodes ) {
                        hid.calculateDelta(teacher);
                    }

                    // for (Node test : hiddenNodes ) {
                    //     System.out.println("delta of hidden: " + test.getDelta());
                    // }


                    // getting loss from an instance in the training set


                   //  double loss = loss(trainingSet.get(j)); 
                   // // trainingSet.get(0).attributes.get(0)

                   //  for (int x = 0; x < outputNodes.size(); x++) {
                   //      if (x == teacher) {
                   //          //loss += (1 * Math.log(outputNodes.get(x).getOutput()));
                   //          loss += (trainingSet.get(j).classValues.get(x) * Math.log(outputNodes.get(x).getOutput()));
                   //          System.out.println(outputNodes.get(x).getOutput());
                   //      }
                   //      else {
                   //          loss += (trainingSet.get(j).classValues.get(x) * Math.log(outputNodes.get(x).getOutput()));
                   //          System.out.println(outputNodes.get(x).getOutput());
                   //      }
                   //  }

                   //  loss = -loss;

                   //  System.out.format("loss : %.8f", + loss);


                    // updating weights for everynode 
                    // probably easier to begin with outputNodes
                    // because from their we can access hidden nodes
                    // so what weights do we have to update?
                    // weights from input to hidden
                    // weights from hidden to output
                    // 

                   // int temp = 0;

                    // testing to see the weight
                    // for (Node out : outputNodes) {
                    //        for (NodeWeightPair n : out.parents ) {
                    //            System.out.println("Weight before updating: " + temp++ + " "  + n.weight);
                    //        }
                    // }



                    // update the weight from hidden nodes to output nodes
                    for (Node out : outputNodes) {
                           out.updateWeight(learningRate);
                    }

                    for (Node hid : hiddenNodes) {
                        hid.updateWeight(learningRate);
                    }



                  //  int tem = 0;
                    // for (Node out : outputNodes) {
                    //     for (NodeWeightPair n : out.parents ) {
                    //            System.out.println("Weight After updating: " + tem++ + " "  + n.weight);
                    //        }
                    // }
                    //calculate error (Tk - Ok) at each output unit k
                  //  for (Node a : outputNodes ) {

                   //     a.calculateDelta(teacher, outputNodes); // the parametmer outputNodes don't really need in here
                   //     System.out.println("delta values: " + a.getDelta());
                   // }

                    // get stuck at calculateDelta
                    //for each hidden unit j and output unit k compute
                   /* for (Node hid : hiddenNodes) { // for each hidden nodes
                        for (Node out : outputNodes  ) { // for each output nodes
                            //α aj (Tk - Ok) g’(ink)
                            // learning rate * outPut of a hidden unit * .
                            deltaWeightHidden = learningRate * hid.getOutput() * out.calculateDelta();
                            // create a new arrayList in here to store the values
                            // Question: Delta is private, how do we get that?
                            // Maybe we have to sohuld some function in Node
                        }
                    }

                    */


                    //for each hidden unit j and output unit k compute
                    // calculate delta weight from hidden to output
                 /*   for (Node out: outputNodes) {
                        out.updateWeight(learningRate);
                    }

                    */

                    // for each input unit i and hidden unit j compute
                    // Δwi,j =α ai Δj
                    // calculate delta weight from input to hidden
               /*     for (Node hid: hiddenNodes) {
                        hid.updateWeight(learningRate);
                    }
            */
                  


            }   // end for                
        }
    }

    public double weightedSum() { //sum of the outputNodes
        
        double temp = 0.0;
        /*
        for (int i = 0; i < outputNodes.size(); i++) { // for all the output Nodes
            Node node = outputNodes.get(i);
            
            
            for (int j = 0; j < node.parents.size(); j++) {
                // total sum equals to weight of hidden nodes * output of those nodes
                sum += (node.parents.get(j).weight * node.parents.get(j).node.getOutput());

            }
        }
        */

        // get all the parents of a node, and then calculate the total sum
        for (Node node : outputNodes ) {
            double sum = 0.0;
            for (NodeWeightPair par : node.parents) {
                sum += (par.weight * par.node.getOutput());
            }
            sum = Math.exp(sum);
            temp += sum;
        }
        return temp;
    }

    /**
     * Calculate the cross entropy loss from the neural network for
     * a single instance.
     * The parameter is a single instance
     */
    private double loss(Instance instance) {
        // TODO: add code here


    	
        return -1.0;
    }
}
