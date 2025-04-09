
public class Main           // SOFTMAX IN OUTPUT LAYER            // Cross Entropy Loss
{                           // si = e^zi / summation(e^zi)        // L = - summation(yi*log(pi))  yi->predicted pi-> true
    public static void main( String[] args )       // dsi/dzi = zi ( 1 - zi) (diagonal)    dsi/dz = -zixzj ( other than diagonal)
    {                                              // softmax deriv combined with cross entropy loss deriv = si - yi
        double[][] matrix = { { 1, 2, 3 },         // https://www.parasdahal.com/softmax-crossentropy
                              { 4, 5, 6 },
                              { 7, 8, 9 } };
    
        double[] vector = { 2,
                            3,
                            1 };

        int[] hidden = { 2 , 3};
        
        NeuralNetwork nn = new NeuralNetwork( vector, hidden, 5);

        System.out.println(nn);

        nn.forwardPass();

        System.out.println(nn);

    }
}