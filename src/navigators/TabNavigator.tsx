import { createBottomTabNavigator } from '@react-navigation/bottom-tabs';
import React, { ReactNode, useEffect } from 'react';
import ExploreNavigator from './ExploreNavigator';
import EventNavigator from './EventNavigator';
import { AddNewScreen, HomeScreen } from '../screens';
import MapNavigator from './MapNavigator';
import ProfileNavigator from './ProfileNavigator';
import { appColors } from '../constants/appColors';
import {
  AddSquare,
  Calendar,
  Home2,
  Iost,
  Location,
  User,
  Bubble
} from 'iconsax-react-native';
import { CircleComponent, TextComponent } from '../components';
import { Platform, View } from 'react-native';
import { globalStyles } from '../styles/globalStyles';
import DrawerNavigator from './DrawerNavigator';
import HomeChatScreen from '../screens/chat/HomeChatScreen';
import ChatApp from '../screens/chat/ChatApp';
import { useNavigation } from '@react-navigation/native';

const TabNavigator = () => {
  const Tab = createBottomTabNavigator();
  const navigation = useNavigation();

  useEffect(() => {
    // Ẩn thanh tab bar khi màn hình được render
    navigation.setOptions({ tabBarVisible: false });
  }, []);
  return (
    <Tab.Navigator
      screenOptions={({ route }) => ({
        headerShown: false,
        tabBarStyle: {
          height: Platform.OS === 'ios' ? 88 : 68,
          justifyContent: 'center',
          alignItems: 'center',
          backgroundColor: appColors.white,
        },
        tabBarIcon: ({ focused, color, size }) => {
          let icon: ReactNode;
          color = focused ? appColors.primary : appColors.gray5;
          size = 24;
          switch (route.name) {
            case 'Khám phá':
              icon = <Home2 size={size} color={color} />;
              break;

            case 'Lịch':
              icon = <Calendar size={size} variant="Bold" color={color} />;
              break;
            case 'Bản đồ':
              icon = <Location size={size} variant="Bold" color={color} />;
              break;
            case 'Hồ sơ':
              icon = <User size={size} variant="Bold" color={color} />;
              break;

            case 'Add':
              icon = (
                <CircleComponent
                  size={52}
                  styles={[
                    globalStyles.shadow,
                    { marginTop: Platform.OS === 'ios' ? -50 : -60 },
                  ]}>
                  <AddSquare size={24} color={appColors.white} variant="Bold" />
                </CircleComponent>
              );
              break;
          }
          return icon;
        },
        tabBarIconStyle: {
          marginTop: 8,
        },
        tabBarLabel({ focused }) {
          return route.name === 'Add' ? null : (
            <TextComponent
              text={route.name}
              flex={0}
              size={12}
              color={focused ? appColors.primary : appColors.gray5}
              styles={{
                marginBottom: Platform.OS === 'android' ? 12 : 0,
              }}
            />
          );
        },
      })}>
      <Tab.Screen name="Khám phá" component={ExploreNavigator} />
      <Tab.Screen name="Lịch" component={EventNavigator} />
      <Tab.Screen name="Add" component={AddNewScreen} />
      <Tab.Screen name="Bản đồ" component={MapNavigator} />
      <Tab.Screen name="Hồ sơ" component={ProfileNavigator} />
    </Tab.Navigator>
  );
};

export default TabNavigator;
