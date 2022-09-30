import jdk.dynalink.Operation;

public class test {
    public static void main(String[] args) {
        String s1="";
        s1=s1+"dsfdsdf";
        System.out.println(s1.length());
        System.out.println(s1);
         s1="(" + s1 + ")";
        System.out.println(s1);
        Operation[] op=new Operation[2];
        int num = Integer.parseInt("8+3-5");
        System.out.println(num);
    }
}
