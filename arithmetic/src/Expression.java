public class Expression {
    public static final String SYMBOL[] = {"+","-","*","÷"};
    private static final int MAXSYMBOL = 3;
    private int n,r;
    //这个表达式生成的二叉树
    private BinaryTree root;

    //计算结果
    private Fraction result;

    //表达式的字符串
    private String expression;
    public Expression(int n,int r){
        this.n=n;
        this.r=r;
    }

    /**
     * 按照限制的操作数随机生成表达式二叉树
     */
    public Expression(int maxNum) throws Exception {
        //随机决定符号数量
        int symbolNum = (int)(Math.random()*MAXSYMBOL)+1;
        root = generateBinaryTree(maxNum,symbolNum);
        result =getResult(root);
        expression = root.midTraversing()+" =";
    }


    /**
    *随机生成二叉树
     */
    public BinaryTree generateBinaryTree(int maxNum, int symbolNum){
        BinaryTree binaryTree = new BinaryTree();
        if(symbolNum==0){
            binaryTree.fraction = NumberFactory.getNumber(maxNum);
        }
        else{
            binaryTree.symbol = SYMBOL[(int)(Math.random() * 4)];
            int leaveSymbolNum = symbolNum-1;
            int symbolNumToLeft = (int)(Math.random()*(leaveSymbolNum+1));
            binaryTree.leftChild = generateBinaryTree(maxNum,symbolNumToLeft);
            binaryTree.rightChild = generateBinaryTree(maxNum,leaveSymbolNum-symbolNumToLeft);
        }
        return binaryTree;
    }


    /***
     * 判断是否需要加括号
     */
    public static boolean addBrackets(String symbol1,String symbol2,int leftOrRight){
        if(symbol2==null){
            return false;
        }
        if(symbol1.equals(SYMBOL[1])){
            if(symbol2.equals(SYMBOL[1])||symbol2.equals(SYMBOL[0])) {
                if (leftOrRight == 2) return true;
            }
        }
        if(symbol1.equals(SYMBOL[2])){
            if(symbol2.equals(SYMBOL[0])||symbol2.equals(SYMBOL[1])){
                return true;
            }
        }
        if(symbol1.equals(SYMBOL[3])){
            if(symbol2.equals(SYMBOL[0])||symbol2.equals(SYMBOL[1])){
                return true;
            }
            if(leftOrRight==2){
                return true;
            }
        }
        return false;
    }
    /**
            * 计算二叉树的结果
     */
    public Fraction getResult(BinaryTree binaryTree) throws Exception {
        if(binaryTree.leftChild==null&&binaryTree.rightChild==null){
            return binaryTree.fraction;
        }
        else{
            String symbol = binaryTree.symbol;
            Fraction leftChildFraction = getResult(binaryTree.leftChild);
            Fraction rightChildFraction = getResult(binaryTree.rightChild);
            binaryTree.fraction = operation(symbol,leftChildFraction,rightChildFraction);
            //若结果为负数，左右子树交换，值取绝对值
            if(binaryTree.fraction.getMolecule()<0){
                binaryTree.fraction.setMolecule(Math.abs(binaryTree.fraction.getMolecule()));
                BinaryTree node = binaryTree.leftChild;
                binaryTree.leftChild = binaryTree.rightChild;
                binaryTree.rightChild = node;
            }
            return binaryTree.fraction;
        }
    }



    /**
     * 左右节点加减乘除计算，返回Fraction
     */
    public Fraction operation(String symbol,Fraction leftChildFraction, Fraction rightChildFraction) throws Exception {
        switch (symbol){
            case "+":
                return Fraction.add(leftChildFraction,rightChildFraction);
            case "-":
                return Fraction.subtraction(leftChildFraction,rightChildFraction);
            case "*":
                return Fraction.multiplication(leftChildFraction,rightChildFraction);
            case "÷":
                return Fraction.division(leftChildFraction,rightChildFraction);
            default:
                System.out.println("error");
                return null;
        }
    }
    /**
     *  重写判断两个Expression对象是否相同的方法
     * @param obj
     * @return
     */
    @Override
    public boolean equals(Object obj) {
        Expression expression = (Expression)obj;
        //先判断表达式的结果是否相同
        if(expression.getResult().toString().equals(this.result.toString())){
            //再判断二叉树是否相同
            if(expression.getRoot().equals(this.root)){
                return true;
            }
        }
        return false;
    }
    public BinaryTree getRoot() {
        return root;
    }

    public void setRoot(BinaryTree root) {
        this.root = root;
    }

    public Fraction getResult() {
        return result;
    }

    public void setResult(Fraction result) {
        this.result = result;
    }

    public String getExpression() {
        return expression;
    }

    public void setExpression(String expression) {
        this.expression = expression;
    }
}
