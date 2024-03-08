import { View, Text } from 'react-native';
import React from 'react';
import { createDrawerNavigator } from '@react-navigation/drawer';
import DrawerCustom from '../components/DrawerCustom';
import TabNavigator from './TabNavigator';
import ChatApp from '../screens/chat/ChatApp';
import HomeScreen from '../screens/home/HomeScreen'
import { createNativeStackNavigator } from "@react-navigation/native-stack";
const Drawer = createDrawerNavigator();
const Stack = createNativeStackNavigator()
const DrawerNavigator = () => {
  return (
    <Drawer.Navigator
      screenOptions={{
        headerShown: false,
        drawerPosition: 'left',
      }}
      drawerContent={props => <DrawerCustom {...props} />}>
      <Drawer.Screen name="HomeNavigator" component={TabNavigator} />
      <Drawer.Screen name="ChatApp" component={ChatApp} />
      <Stack.Screen name='HomeScreen' component={HomeScreen} />
    </Drawer.Navigator>
  );
};

export default DrawerNavigator;