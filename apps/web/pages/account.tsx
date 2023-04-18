'use client';
import {
  Box,
  Text,
  Button,
  Flex,
  useToast,
  Divider,
  chakra,
} from '@chakra-ui/react';
import Layout from '@components/Layout';
import React, { useState, useEffect } from 'react';
import { signOut } from 'next-auth/react';
import { getServerSession } from 'next-auth/next';
import { authOptions } from './api/auth/[...nextauth]';
import { ReactElement } from 'react';
import { trpc } from '@utils/trpc';
import { useRouter } from 'next/router';
import { TiTick } from 'react-icons/ti';
import { RxCross2 } from 'react-icons/rx';
import AccountGameplayItem from '@components/account/AccountGameplayItem';
import DeleteAccModal from '@components/account/DeleteAccModal';
import Head from 'next/head';
import { FaDiscord, FaBattleNet, FaTwitch } from 'react-icons/fa';
import { FcGoogle } from 'react-icons/fc';
import { BsGithub } from 'react-icons/bs';
import { MdOutlineRemove } from 'react-icons/md';
import useSite from '@site';
import { prisma } from '@server/db/client';
import Loading from '@components/Loading';
import { GetServerSideProps } from 'next';
import * as Sentry from '@sentry/nextjs';

type ProvidersListType = {
  name: string;
  icon: React.ReactElement;
};

const ProvidersList: Array<ProvidersListType> = [
  {
    name: 'Discord',
    icon: <FaDiscord size={30} color={'#5865F2'} />,
  },
  {
    name: 'Google',
    icon: <FcGoogle size={30} />,
  },
  {
    name: 'BattleNet',
    icon: <FaBattleNet size={30} color={'#009AE4'} />,
  },
  {
    name: 'Twitch',
    icon: <FaTwitch size={30} color={'#9146FF'} />,
  },
  {
    name: 'Github',
    icon: <BsGithub size={30} color={'#000000'} />,
  },
];

export interface Gameplay {
  id: string;
  userId: string;
  youtubeUrl: string;
  gameplayType: string;
  isAnalyzed: boolean;
}

export default function Account() {
  const { session, isLoading, setLoading } = useSite();
  const utils = trpc.useContext();
  const router = useRouter();
  const toast = useToast();
  // Account Providers logic
  const { isLoading: isProvidersDataStillFetching, data: providersData } =
    trpc.user.getLinkedAccounts.useQuery();

  // States
  const [linkedAccounts, setLinkedAccounts] = useState(providersData);
  const [showModal, setShowModel] = useState(false);

  // Unlink a users non primary account
  const unlinkAccount = trpc.user.unlinkAccount.useMutation({
    async onSuccess() {
      await utils.user.invalidate();
    },
  });
  // Show modal handler
  const handleAccountDeletion = async () => {
    setShowModel(!showModal);
  };
  // Unlink a users non primary account handler
  const unlinkProvider = async (account: {
    id: string;
    userId: string;
    provider: string;
  }) => {
    try {
      await unlinkAccount.mutateAsync({ accountId: account.id });
      toast({
        position: 'bottom-right',
        title: 'Unlink Account',
        description: 'Successfully unlinked the account.',
        status: 'success',
        duration: 5000,
        isClosable: true,
      });
    } catch (error) {
      Sentry.captureException(error);
      toast({
        position: 'bottom-right',
        title: 'Unlink Account',
        description: 'An unexpected error occurred, please try again later.',
        status: 'error',
        duration: 5000,
        isClosable: true,
      });
    }
  };

  // Gameplay list logic
  const { isLoading: isGameplayDataStillFetching, data: gamplayData } =
    trpc.gameplay.getUsers.useQuery({
      userId: undefined,
    });

  // States
  const [gameplayItems, setGameplayItems] = useState<
    Array<Gameplay> | undefined
  >();

  useEffect(() => {
    if (
      !isProvidersDataStillFetching &&
      !isGameplayDataStillFetching &&
      session?.user?.name
    ) {
      setLinkedAccounts(providersData);
      setGameplayItems(gamplayData);
      setLoading(false);
    }
  }, [
    isLoading,
    providersData,
    isProvidersDataStillFetching,
    router,
    session,
    setLoading,
    isGameplayDataStillFetching,
    gamplayData,
  ]);

  return isLoading ? (
    <Loading color={'purple.500'} />
  ) : (
    <>
      <Head>
        <title>Waldo Vision | Account</title>
        <meta
          name="description"
          content="Waldo Vision is an open-source visual cheat detection project, powered by deep learning"
        />
      </Head>
      <Box minHeight={'100vh'} mt={{ base: '60px' }} mb={20}>
        <DeleteAccModal show={showModal} />
        <Flex direction={'column'} gap={3}>
          <Flex direction={'column'}>
            <Text fontWeight={'bold'} fontSize={30}>
              Your Account
            </Text>

            <Text fontWeight={'normal'} fontSize={15}>
              Manage your account settings.
            </Text>
          </Flex>
          <Box
            borderRadius={'16px'}
            bg={'white'}
            p={{ base: 4, md: 8 }}
            overflow={'hidden'}
            minWidth={'30vw'}
            maxWidth={'7xl'}
            minHeight={'60vh'}
            width={{ base: '90vw', md: '80vw' }}
          >
            <Flex direction={'column'} gap={20}>
              <Flex
                direction={{ base: 'column', md: 'row' }}
                justify={'space-between'}
                gap={10}
              >
                {/* Linked Account */}
                <Box width={{ md: '100%', lg: '50%' }}>
                  <Flex gap={2} direction={'column'}>
                    <Text
                      fontWeight={'normal'}
                      mb={5}
                      fontSize={{ base: 15, sm: 18 }}
                    >
                      You have linked <b>all</b> of the following accounts:
                    </Text>
                    {(session?.user?.role == 'ADMIN' ||
                      session?.user?.role == 'MOD') && (
                      <Text>
                        Redirect me to the{' '}
                        <chakra.span
                          fontWeight={'bold'}
                          textDecor={'underline'}
                          cursor={'pointer'}
                          onClick={() => router.push('/dash/users')}
                        >
                          Admin Dashboard
                        </chakra.span>
                      </Text>
                    )}
                  </Flex>
                  <Flex direction={'column'}>
                    {linkedAccounts &&
                      ProvidersList.map(({ name, icon }, index) => (
                        <Box key={index}>
                          <Divider my={5} />
                          <Flex direction={'column'} gap={2}>
                            <Flex align={'center'} direction={'row'} gap={3}>
                              {icon}
                              <Text fontWeight={500}>{name}</Text>
                            </Flex>
                            <Flex direction={'column'}>
                              {linkedAccounts.some(
                                account =>
                                  account.provider.toUpperCase() ===
                                  name.toUpperCase(),
                              ) ? (
                                <>
                                  {name.toUpperCase() ==
                                  session?.user?.provider.toUpperCase() ? (
                                    <Flex
                                      direction={'row'}
                                      align={'center'}
                                      ml={3}
                                      gap={1}
                                    >
                                      <TiTick color={'green'} />
                                      <Text fontSize={{ base: 13, sm: 15 }}>
                                        This is your primary account
                                      </Text>
                                    </Flex>
                                  ) : (
                                    <></>
                                  )}

                                  <Flex
                                    direction={'row'}
                                    align={'center'}
                                    ml={3}
                                    gap={1}
                                  >
                                    <TiTick color={'green'} />
                                    <Text
                                      fontSize={{
                                        base: 11,
                                        sm: 13,
                                        md: 15,
                                      }}
                                    >
                                      You are connected to a{' '}
                                      {name.toLowerCase()} account.
                                    </Text>
                                  </Flex>
                                  <Flex direction={'column'}>
                                    <Divider my={5} />
                                    <Text fontWeight={'bold'}>API Access</Text>
                                    <Text
                                      fontWeight={'regular'}
                                      mb={5}
                                      fontSize={15}
                                      _hover={{ textDecoration: 'underline' }}
                                      cursor={'pointer'}
                                      onClick={() =>
                                        router.push(
                                          '/portal/developers/apikeys',
                                        )
                                      }
                                    >
                                      <chakra.span
                                        fontWeight={'bold'}
                                        textColor={'purple.500'}
                                      >
                                        Click
                                      </chakra.span>{' '}
                                      to go to the developer portal.
                                    </Text>
                                  </Flex>
                                  {name.toUpperCase() !=
                                  session?.user?.provider.toUpperCase() ? (
                                    <Button
                                      leftIcon={<MdOutlineRemove />}
                                      colorScheme={'red'}
                                      variant={'solid'}
                                      my={3}
                                      onClick={() => {
                                        const account = linkedAccounts.find(
                                          account =>
                                            account.provider.toUpperCase() ===
                                            name.toUpperCase(),
                                        );
                                        if (!account) {
                                          toast({
                                            position: 'bottom-right',
                                            title: 'Account Error',
                                            description:
                                              'An unexpected error occurred, please try again later.',
                                            status: 'error',
                                            duration: 5000,
                                            isClosable: true,
                                          });
                                          return;
                                        }
                                        unlinkProvider(account);
                                      }}
                                    >
                                      Disconnect
                                    </Button>
                                  ) : (
                                    <></>
                                  )}
                                </>
                              ) : (
                                <Flex
                                  direction={'row'}
                                  align={'center'}
                                  ml={3}
                                  gap={1}
                                >
                                  <RxCross2 color={'red'} />
                                  <Text fontSize={{ base: 11, sm: 13, md: 15 }}>
                                    You are not connected to a{' '}
                                    {name.toLowerCase()} account.
                                  </Text>
                                </Flex>
                              )}
                            </Flex>
                          </Flex>
                        </Box>
                      ))}
                  </Flex>
                </Box>
                {/* Uploaded Gameplay */}
                <Box width={{ md: '100%', lg: 'fit' }} height={'100%'}>
                  <Flex
                    direction={'column'}
                    gap={10}
                    maxW={'100%'}
                    minHeight={'100%'}
                    maxHeight={'2xl'}
                    overflowY={'auto'}
                  >
                    {gameplayItems &&
                      gameplayItems?.map((item, index) => (
                        <AccountGameplayItem key={index} item={item} />
                      ))}
                  </Flex>
                </Box>
              </Flex>
              {/* Delete Account */}
              <Flex
                direction={{ base: 'column', md: 'row' }}
                ml={{ md: 'auto' }}
                right={0}
                gap={3}
              >
                <Button
                  bgColor={'red.300'}
                  color={'white'}
                  variant={'solid'}
                  _hover={{ bgColor: 'white', color: 'red.400' }}
                  onClick={() => handleAccountDeletion()}
                >
                  Delete Account
                </Button>
                <Button
                  bgColor={'#373737'}
                  backgroundColor={'gray.800'}
                  colorScheme={'blackAlpha'}
                  boxShadow={'lg'}
                  onClick={() => router.push('/auth/login')}
                >
                  Add Connection
                </Button>
                <Button
                  bgColor={'#373737'}
                  backgroundColor={'gray.800'}
                  colorScheme={'blackAlpha'}
                  boxShadow={'lg'}
                  onClick={() => signOut()}
                >
                  Logout
                </Button>
              </Flex>
            </Flex>
          </Box>
        </Flex>
      </Box>
    </>
  );
}

export const getServerSideProps: GetServerSideProps = async ({ req, res }) => {
  const user = await getServerSession(req, res, authOptions);
  const config = await prisma.waldoPage.findUnique({
    where: {
      name: 'account',
    },
  });

  if (config && config?.maintenance) {
    return { redirect: { destination: '/', permanent: false }, props: {} };
  } else if (!user)
    return { redirect: { destination: '/auth/login' }, props: {} };
  else
    return {
      props: {},
    };
};

Account.getLayout = function getLayout(page: ReactElement) {
  return <Layout>{page}</Layout>;
};
