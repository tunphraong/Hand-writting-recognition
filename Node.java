import java.util.*;

/**
 * Class for internal organization of a Neural Network.
 * There are 5 types of nodes. Check the type attribute of the node for details.
 * Feel free to modify the provided function signatures to fit your own implementation
 */

public class Node {
    private int type = 0; //0=input,1=biasToHidden,2=hidden,3=biasToOutput,4=Output
    public ArrayList<NodeWeightPair> parents = null; //Array List that will contain the parents (including the bias node) with weights if applicable

    private double inputValue = 0.0;
    private double outputValue = 0.0;
    private double outputGradient = 0.0;
    private double delta = 0.0; //input gradient

    //Create a node with a specific type
    Node(int type) {
        if (type > 4 || type < 0) {
            System.out.println("Incorrect value for node type");
            System.exit(1);

        } else {
            this.type = type;
        }

        if (type == 2 || type == 4) {
            parents = new ArrayList<>();
        }
    }

    //For an input node sets the input value which will be the value of a particular attribute
    public void setInput(double inputValue) {
        if (type == 0) {    //If input node
            this.inputValue = inputValue;
        }
    }

     public void setGradient(double gradient) {
        
            this.outputGradient = gradient;
        
    }

    /**
     * Calculate the output of a node.
     * You can get this value by using getOutput()
     */
        public void calculateOutput(double sumOfOutput) {
        if (type == 2 || type == 4) {   //Not an input or bias node
            // TODO: add code here

            // how do we calculate Output in here for hidden and output nodes?
            // equal to the weight * attributes?
            // calculates the output at the current node and stores that value in a member 
            // variable called output value

            // calculate value of a node. 
            // equals to the sum of all the parents: weights * input?
            
           //so if the node is type 2, we need to use 
           // reLU function : g(𝑧) = max(0, 𝑧)

           // we only need to update this.outputValue
           // but how do we know what node to calculate?

            double sum = 0.0;

            if (type == 2) { // type 2: hidden use ReLU
                // for (int i = 0; i < parents.size() ;i++) {
                //     sum += (parents.get(i).weight * parents.get(i).node.outputValue);            
                // }

                for (NodeWeightPair hid : parents ) {
                    sum += (hid.weight * hid.node.getOutput());
                }
                outputValue = Math.max(0, sum);
               // System.out.println("output of hidden nodes:"  + outputValue + "sum" + sum) ;
            }
            else if (type == 4) { //Output -> use Softmax function

               //  assume that we know all the outputs of the hidden units
               // calculate the output of output nodes
                for (int i = 0; i < parents.size() ;i++) {
                    sum += (parents.get(i).weight * parents.get(i).node.getOutput());            
                }

                double output = Math.exp(sum) / sumOfOutput;

                this.outputValue = output;
            }
        }
    }

    //Gets the output value
    public double getOutput() {

        if (type == 0) {    //Input node
            return inputValue;
        } else if (type == 1 || type == 3) {    //Bias node
            return 1.00;
        } else {
            return outputValue;
        }
    }

    public double getDelta() {

       // if (type == 2 || type == 4)
            return this.delta;
    }

    //Calculate the delta value of a node.
    // store in delta
    //type 2 means ReLU, type 4 means Softmax
    public void calculateDelta(int teacher) {
        if (type == 2 || type == 4)  {
            if (type == 4) { // if the node is output
                //yj - g(zj)?
                //yj is the teacher
                // g(zj) is the softmax function 
                double delta = teacher - outputValue;
                //System.out.println("teacher: " + teacher);
                this.delta = delta;
            }
            
            if (type == 2)  { // hidden unit
            	// we want to calculate the delta of hidden unit

            	double delta = 0.0;
            	double sum = 0.0;
            	int gPrime = 0;
            	
            	//calculate z 
            	// for (int i = 0; i < parents.size() ;i++) {
             //        z += (parents.get(i).weight * parents.get(i).node.outputValue);
             //    }
                for (NodeWeightPair hid : parents ) {
                    sum += (hid.weight * hid.node.getOutput());
                }

                 //weight of jk * delta k
                //come up with gPrime
            	if (sum <= 0) gPrime = 0;
                else gPrime = 1;

                this.delta = gPrime * outputGradient;

                // sum of all the output nodes
                // weight of jk * delta k
                // k is the output
                // assume we already know delta k
                // how do get accessed to the outputNodes  
             }    
        }   
    }
    //Update the weights between parents node and current node
    public void updateWeight(double learningRate) {
        if (type == 2 || type == 4) {
            // TODO: add code here
            // udpate parents weight
            // calculate delta weight

            for (NodeWeightPair a : parents) {
               //Δwj,k = α aj ∆k
               double deltaWeight = learningRate * a.node.getOutput() * this.delta;
               a.weight += deltaWeight;
            }
            // do we need to do w + delta weight in here?
        }
    }

    public void normalizeOutput(double sum) {
            if (this.type == 4) {
                this.outputValue /= sum;
            }
    }
}


