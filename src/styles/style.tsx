import { ImageBackground, StyleSheet } from "react-native";
import LinearGradient from 'react-native-linear-gradient';

interface Colors {
  primary: string;
  secondary: string;
  tertiary: string;
  alternative: string;
  fb: string;
  disabled: string;
  userMessageBackground: string;
}
export const colors: Colors = {
  primary: "#fff",
  secondary: "#adadad",
  tertiary: "#057afd",
  alternative: "#666",
  fb: "#39559f",
  disabled: "rgba(5, 122, 253, 0.5)",
  userMessageBackground: "#3081D0"
};

export const basic = StyleSheet.create({
  gradientBox: {
    flex: 1,
    width: '100%',
    height: 500,
  },
  linearGradient: {
    flex: 1,
    paddingLeft: 15,
    paddingRight: 15,
    borderRadius: 5
  },
});

export const center = StyleSheet.create({
  containerBox1: {
    marginTop: 90,
    backgroundColor: '#FCFDFE',
    borderRadius: 20,
    flexDirection: 'column',
    justifyContent: 'center',
    marginHorizontal: 80,
  },
  containerBox2: {
    marginTop: 10,
    backgroundColor: '#FCFDFE',
    borderRadius: 20,
    flexDirection: 'column',
    justifyContent: 'center',
    marginHorizontal: 70,
    padding: 5,
  },
  containerBox3: {
    marginTop: 30,
    backgroundColor: '#FCFDFE',
    borderRadius: 20,
    flexDirection: 'column',
    justifyContent: 'center',
    marginHorizontal: 80,
    padding: 5,
  },
  containerBox4: {
    marginTop: 30,
    backgroundColor: '#FCFDFE',
    borderRadius: 20,
    flexDirection: 'column',
    justifyContent: 'center',
    marginHorizontal: 80,
    padding: 5,
  },
  text: {
    color: '#000000',
    textAlign: 'center',
    fontSize: 16,
  },
  dot: {
    fontSize: 20,
    color: "#BF95E4",
    textAlign: "center",
    textAlignVertical: "top",
  },
  dot2: {
    fontSize: 20,
    color: "#55B9AC",
    textAlign: "center",
    textAlignVertical: "top",
  },
  icon: {
    alignSelf: 'center',
    top: 80,
  },
  icon1: {
    alignSelf: 'center',
    top: 30,
  },
  icon2: {
    alignSelf: 'center',
    top: 25,
  }
});
export const form = StyleSheet.create({
  input: {
    flex: 1,
    fontWeight: "bold",
    letterSpacing: 1,
    fontSize: 15,
    textAlign: "left",
    marginLeft: 5,
    overflow: 'hidden',
    backgroundColor: '#E7E9EB',
    borderRadius: 30,
    paddingLeft: 10,
    color: '#000'

  },
  message: {
    textAlign: "center",
    fontSize: 20,
    color: "tomato"
  },
  messageContainer: {
    padding: 8,
    borderRadius: 10,
    marginVertical: 5,
    maxWidth: '80%',
    alignSelf: 'flex-end',
    backgroundColor: colors.userMessageBackground,
  },
  messageText: {
    fontSize: 30,
    color: '#ffffff',
  },
  friendMessageContainer: {
    alignSelf: 'flex-start',
    backgroundColor: '#EBECF0',
    borderRadius: 10,
    padding: 10,
    marginVertical: 4,
    maxWidth: '70%',
  },
  iconsContainerLink: {
    flexDirection: "row",
    alignItems: "center",
    backgroundColor: '#F0F4FD',
    borderRadius: 5,
    paddingHorizontal: 10,
  },
  iconLink: {
    margin: 5,
  },
  iconVoice: {
    margin: 5,
  },
  iconSend: {
    margin: 5, // Thêm flex: 1 để các icon tự động căn chỉnh
  },
  topContainer: {
    backgroundColor: 'F0F4FD',
    flexDirection: "row",
    justifyContent: "space-between",
  },
  iconBack: {
    top: 10,
    alignSelf: 'flex-start',
    marginLeft: 7,
  },
  iconInf: {
    top: 10,
    alignSelf: 'flex-end',
    marginRight: 7,
  },
  header: {
    marginTop: 30,
    alignItems: "center",
    justifyContent: "center",
  },
  title: {
    textAlign: 'center',
    fontSize: 40,
    fontWeight: "600",
    color: "#000000",
  },
});
