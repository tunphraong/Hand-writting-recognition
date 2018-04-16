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

    public void fp(ArrayList<Double> a) {
            for (int i = 0; i < a.size(); i++) {
                this.inputNodes.get(i).setInput(a.get(i));
            }
            for (int j = 0; j < hiddenNodes.size(); j++) {
                hiddenNodes.get(j).calculateOutput(weightedSum());
            }
            double sum = 0.0;
            for (int k = 0; k < outputNodes.size(); k++) {
                outputNodes.get(k).calculateOutput(weightedSum());
                sum = outputNodes.stream().mapToDouble(Node::getOutput).sum(); //double s = outputNodes.stream().mapToDouble(Node::getOutput).sum();
            }
             for (int k = 0; k < outputNodes.size(); k++) {
                 outputNodes.get(k).normalizeOutput(sum);
             }
    }

    // public int predict(Instance instance) { // ultimately, we want output
    //     // TODO: add code here
    //     // we are given one instance
    //     // instance contains an array of attributes
    //     // an attribute contains an array of input values -> so they are # of input?



    //   // for (Node in : inputNodes ) {
    //    //     System.out.println("output of input nodes:" + in.getOutput());
    //     //}

    //     // we need to pass in total outputSums to calculateOutput function

    //     // calculate output for hidden nodes

   

    //     //System.out.println(temp);

    //     // given input for input nodes, we can calculate values of hidden nodes
    //    // for (int j = 0;j < hiddenNodes.size();j++) {
    //     //    hiddenNodes.get(j).getOutput();
    //    // }

    //    // for (int k = 0; k < outputNodes.size();k++) {
    //    //     outputNodes.get(k).getOutput();; // get nodes and calculate output 
    //    // }

    //     fp(instance.attributes);

    //     int ret = 0;
    //     double temp = Double.MIN_VALUE;
    //   //  System.out.println("output of first: " + temp);

    //     for (int j = 1; j < outputNodes.size(); j++) {
    //         if (temp < outputNodes.get(j).getOutput()) {
    //             temp = outputNodes.get(j).getOutput();
    //             ret = j;
    //         }
    //     }

    //    // System.out.println(ret);

    //     return ret;
    // }

     public int predict(Instance instance) {
        // TODO: add code here
           
            
            fp(instance.attributes);
            double gs = Double.MIN_VALUE;
            int lab = 0;
            
            for (int i = 0; i < instance.classValues.size(); i++) {
                if (outputNodes.get(i).getOutput() > gs) {
                    gs = outputNodes.get(i).getOutput();
                    lab = i;
                }
            }          
            return lab;
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

                     // System.out.println("Size of the attributes" +  instance.attributes.size());
                      for (int p = 0; p < trainingSet.get(j).attributes.size(); p++) { // get the size of an instance
                          // set the input values for input nodes
                          inputNodes.get(p).setInput(trainingSet.get(j).attributes.get(p));
                          //System.out.println("instance of input: " + instance.attributes.get(i));
                      }       

                       //  System.out.println("Size of the hidden nodes" +  hiddenNodes.size());
                          for (Node hid: hiddenNodes ) {
                              hid.calculateOutput(weightedSum());
                          }
                  
                  
                           //double temp = 0.0;
                          for (Node out : outputNodes ) {
                              out.calculateOutput(weightedSum());  
                              //temp += out.getOutput();
                          }                     


                    //int outPut = predict(trainingSet.get(j)); // calculate outPut for output nodes
                    // update gradient

                    // get teacher output
                    int teacher = 0;

                    for (int k = 0; k < trainingSet.get(0).classValues.size();k++) {
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

                    for (int n = 0; n < hiddenNodes.size(); n++ ) {
                        
                        double gradient = 0.0;
                    

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

                   // int temp = 0;



                    // update the weight from hidden nodes to output nodes
                    for (Node out : outputNodes) {
                           out.updateWeight(learningRate);
                    }

                    for (Node hid : hiddenNodes) {
                        hid.updateWeight(learningRate);
                    }


                   
                  


            }   // end for



            double totalError = 0;
            for(Instance instance:trainingSet) {
                totalError+=loss(instance);
            }

            System.out.print("Epoch: " + i + ", Loss: ");
            System.out.format("%.8e\n", totalError/trainingSet.size());                
        } // end epoch

        // for (Node out : outputNodes ) {
        //     for (NodeWeightPair par: out.parents) {
        //         System.out.println("WEIGHT from hidden to output"  + par.weight);
        //     }
        // }


        //  // weight from input to hidden
        // for (int temp = 0; temp < hiddenNodes.size();temp++ ) {
        //     for (int weight = 0; weight < 3;weight++ ) {
        //         System.out.println(hiddenNodes.get(temp).parents.get(weight).weight);
        //     }
        // }





        //          for (Node hid: hiddenNodes ) {
        //     System.out.println(hid.getOutput());
        // }

    }

    public double weightedSum() { //sum of the outputNodes
        
        double temp = 0.0;

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
        fp(instance.attributes);
        double t = outputNodes.get(instance.classValues.indexOf(1)).getOutput(); 
        
     return -Math.log(t) ;
    	
    }
}
