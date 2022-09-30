import java.util.Random;

public class NumberFactory {
    /**
     * 生成随机自然数
     */
    public static Fraction getNumber(int max){
        //决定生成分数还是整数
        int isFraction = new Random().nextInt(100);
        if(isFraction>80){
            return Fraction.newFraction(max);
        }
        else{
            int num = (int)(Math.random()*max);
            return Fraction.intChangeToFraction(num);
        }
    }
}
