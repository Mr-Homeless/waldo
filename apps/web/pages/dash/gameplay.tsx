import {
  Box,
  Text,
  Input,
  Button,
  Menu,
  MenuButton,
  MenuList,
  MenuItem,
  Center,
  InputRightElement,
  InputGroup,
  Table,
  Tbody,
  Td,
  Th,
  Thead,
  Tr,
  Flex,
  Tfoot,
  useToast,
  Spinner,
  Link,
} from '@chakra-ui/react';
import Layout from '@components/dashboard/Layout';
import { ReactElement, useEffect, useState } from 'react';
import {
  ChevronDownIcon,
  ChevronLeftIcon,
  ChevronRightIcon,
  SearchIcon,
} from '@chakra-ui/icons';
import { trpc } from '@utils/trpc';
import Loading from '@components/Loading';
import { getServerSession } from 'next-auth/next';
import { BiBlock } from 'react-icons/bi';
import { authOptions } from 'pages/api/auth/[...nextauth]';
import { games } from '@config/gameplay';
import { GameplayType, GameplayTypeWithNull } from '@utils/zod/gameplay';
import { GetServerSideProps } from 'next';
type Query =
  | {
      gameplayCount?: number | undefined;
      id: string;
      userId: string;
      youtubeUrl: string;
      gameplayType: GameplayType;
      isAnalyzed: boolean;
      user?: {
        name?: string | null;
        image?: string | null;
      };
    }[]
  | undefined;
interface GoToItem {
  number: number;
}
export default function Gameplay() {
  // Searching states

  const [searchRole, setSearchRole] = useState<GameplayTypeWithNull>(null);
  // Data and Rows
  // const { data, isLoading } = trpc.user.getUsers.useQuery({ page: 1 });
  const [gtMenuItems, setGtMenuItems] = useState<GoToItem[]>();

  const [pageNumber, setPageNumber] = useState<number>(1);
  const [data, setData] = useState<Query>();
  const {
    data: userQueryData,
    isLoading: userQueryLoading,
    refetch: userQueryRefetch,
  } = trpc.gameplay.getMany.useQuery(
    {
      page: pageNumber,
      filterGames: searchRole,
    },
    { enabled: false },
  );

  const handleFilter = async (role: GameplayTypeWithNull) => {
    if (role == null) {
      setSearchRole(null);
      return;
    }
    setSearchRole(role);
  };

  const handlePageChange = async (add: boolean) => {
    await userQueryRefetch();
    if (add) {
      setPageNumber(pageNumber + 1);
      await userQueryRefetch();
    } else {
      setPageNumber(1);
      await userQueryRefetch();
    }
  };
  useEffect(() => {
    const doLoadThings = async () => {
      await userQueryRefetch();
      setData(userQueryData);
      console.log(userQueryData);
    };
    doLoadThings();
  }, [userQueryData, userQueryRefetch]);
  if (userQueryLoading) {
    return (
      <Box>
        <Loading color={'purple.500'} />
      </Box>
    );
  } else {
    const handlePage = () => {
      if (!data || !data[0].gameplayCount) return;
      if (pageNumber == Math.ceil(data[0].gameplayCount / Math.round(10))) {
        return;
      } else {
        setPageNumber(pageNumber + 1);
      }
    };
    const goToPage = (number: number) => {
      setPageNumber(number);
    };

    const autoFillMenu = () => {
      const menuItems: GoToItem[] = [];
      if (data == undefined || data[0] == undefined) return;
      if (!data[0].gameplayCount) return;
      const totalPages = Math.ceil(data[0].gameplayCount / Math.round(10));
      for (let i = 0; i < totalPages; i++) {
        menuItems.push({ number: i + 1 });
      }
      setGtMenuItems(menuItems);
    };

    return (
      <Center width={'100%'} flexDirection={'column'} gap={5}>
        <InputGroup width={{ base: '100%', md: '60%' }}>
          <Input
            bgColor={'white'}
            borderRadius={16}
            border={'none'}
            height={'50px'}
            fontWeight={'medium'}
            placeholder={'Search Gameplay'}
            disabled={true}
          />
          <InputRightElement mt={1}>
            <SearchIcon />
          </InputRightElement>
        </InputGroup>
        <Box width={{ base: '100%', md: '70%' }}>
          <Flex alignItems={'center'} gap={3}>
            <Menu>
              <MenuButton
                as={Button}
                bgColor={'white'}
                _hover={{ bgColor: 'white' }}
                _active={{ bgColor: 'white' }}
                rightIcon={<ChevronDownIcon />}
              >
                Filters: {searchRole}
              </MenuButton>
              <MenuList>
                {games.map(game => {
                  return (
                    <MenuItem
                      onClick={() =>
                        handleFilter(
                          game.shortName.toUpperCase() as GameplayTypeWithNull,
                        )
                      }
                    >
                      {game.shortName.toUpperCase()}
                    </MenuItem>
                  );
                })}
              </MenuList>
            </Menu>
            <Text fontWeight={'semibold'}>
              Total gameplay items: {data && data[0] && data[0].gameplayCount}
            </Text>
          </Flex>
          <Box overflowX="auto">
            {data && data[0] && (
              <Table
                width={'100%'}
                variant={'simple'}
                sx={{
                  borderCollapse: 'separate',
                  borderSpacing: '0 1rem',
                }}
              >
                <Thead>
                  <Tr bgColor={'white'} height={'50px'}>
                    <Th borderLeftRadius={16}>
                      <Text
                        casing={'capitalize'}
                        fontWeight={'bold'}
                        fontSize={15}
                      >
                        FootageType
                      </Text>
                    </Th>
                    <Th>
                      <Text
                        casing={'capitalize'}
                        fontWeight={'bold'}
                        fontSize={15}
                      >
                        Gameplay Id
                      </Text>
                    </Th>
                    <Th>
                      <Text
                        casing={'capitalize'}
                        fontWeight={'bold'}
                        fontSize={15}
                      >
                        Gameplay URL
                      </Text>
                    </Th>
                    <Th>
                      <Text
                        casing={'capitalize'}
                        fontWeight={'bold'}
                        fontSize={15}
                      >
                        Analyzed
                      </Text>
                    </Th>
                    <Th>
                      <Text
                        casing={'capitalize'}
                        fontWeight={'bold'}
                        fontSize={15}
                      >
                        Gameplay Owner
                      </Text>
                    </Th>
                    <Th borderRightRadius={16}>
                      <Text
                        casing={'capitalize'}
                        fontWeight={'bold'}
                        fontSize={15}
                      >
                        Actions
                      </Text>
                    </Th>
                  </Tr>
                </Thead>
                <Tbody>
                  {userQueryLoading ? (
                    <Tr>
                      <Td></Td>
                      <Td></Td>
                      <Td></Td>
                      <Td>
                        <Center h={600}>
                          <Spinner color={'purple.500'} size={'xl'} />
                        </Center>
                      </Td>
                      <Td></Td>
                    </Tr>
                  ) : (
                    data?.map((result, index) => {
                      return (
                        <Tr bgColor={'white'} height={'70px'} key={index}>
                          <Td borderLeftRadius={16}>
                            <Flex direction={'row'} align={'center'} gap={2}>
                              <Text fontWeight={'bold'}>
                                {result && result.gameplayType}
                              </Text>
                            </Flex>
                          </Td>
                          <Td>
                            <Text fontSize={15}>
                              {result.id && result.id.toLowerCase()}
                            </Text>
                          </Td>
                          <Td>
                            <Link
                              fontSize={15}
                              href={result.youtubeUrl}
                              textColor={'blue.500'}
                            >
                              {result.youtubeUrl}
                            </Link>
                          </Td>
                          <Td>
                            <Text fontSize={15}>
                              {result.isAnalyzed.toString().toUpperCase()}
                            </Text>
                          </Td>
                          <Td>
                            <Text fontSize={15} isTruncated>
                              {result.user && result.user.name}
                            </Text>
                          </Td>
                          <Td borderRightRadius={16}>
                            <MenuAction gameplayId={result.id} />
                          </Td>
                        </Tr>
                      );
                    })
                  )}
                </Tbody>
                {data && data[0] && (
                  <Tfoot bgColor={'white'} height={'50px'}>
                    <Tr>
                      <Td borderLeftRadius={16} />
                      <Td />
                      <Td />
                      <Td />
                      <Td />
                      <Td bgColor={'white'} borderRadius={16} isTruncated>
                        <Flex
                          direction={'row'}
                          align={'center'}
                          textAlign={'center'}
                          gap={2}
                        >
                          <ChevronLeftIcon
                            cursor={'pointer'}
                            h={6}
                            w={6}
                            _hover={{ color: 'gray.400' }}
                            color={pageNumber === 1 ? 'gray.300' : ''}
                            onClick={() =>
                              pageNumber === 1 ? null : handlePageChange(false)
                            }
                          />
                          <Flex>
                            <Text fontWeight={'semibold'}>Page</Text>{' '}
                            <Text ml={2}>
                              {pageNumber} of{' '}
                              {data[0] &&
                                data[0].gameplayCount &&
                                Math.ceil(
                                  data[0].gameplayCount / Math.round(10),
                                )}
                            </Text>
                          </Flex>
                          <ChevronRightIcon
                            cursor={'pointer'}
                            h={6}
                            w={6}
                            _hover={{ color: 'gray.400' }}
                            color={
                              data[0] &&
                              data[0].gameplayCount &&
                              pageNumber ==
                                Math.ceil(
                                  data[0].gameplayCount / Math.round(10),
                                )
                                ? 'gray.300'
                                : ''
                            }
                            onClick={() => handlePage()}
                          />
                        </Flex>
                      </Td>
                    </Tr>
                  </Tfoot>
                )}
              </Table>
            )}
            <Menu>
              <MenuButton
                as={Button}
                bgColor={'white'}
                _hover={{ bgColor: 'white' }}
                _active={{ bgColor: 'white' }}
                rightIcon={<ChevronDownIcon />}
                onClick={() => autoFillMenu()}
              >
                Go to Page
              </MenuButton>
              <MenuList>
                {gtMenuItems &&
                  gtMenuItems.map((item: GoToItem) => {
                    return (
                      <MenuItem onClick={() => goToPage(item.number)}>
                        {item.number}
                      </MenuItem>
                    );
                  })}
              </MenuList>
            </Menu>
          </Box>
        </Box>
      </Center>
    );
  }
}
interface MenuActionProps {
  gameplayId: string;
}
const MenuAction = (props: MenuActionProps) => {
  const gameplayId = props.gameplayId;
  const utils = trpc.useContext();
  const toast = useToast();
  const deleteGameplay = trpc.gameplay.delete.useMutation({
    async onSuccess() {
      await utils.user.invalidate();
    },
  });
  const handleGameplayDeletion = async () => {
    await deleteGameplay.mutateAsync({ gameplayId: gameplayId });
    toast({
      position: 'bottom-right',
      title: 'Gameplay Deletion',
      description: `Successfully deleted the gameplay.`,
      status: 'success',
      duration: 5000,
      isClosable: true,
    });
  };

  return (
    <Menu>
      <MenuButton as={Button} rightIcon={<ChevronDownIcon />}>
        Actions
      </MenuButton>
      <MenuList p={0} borderBottomRadius={12}>
        <MenuItem
          icon={
            <BiBlock color={'white'} size={16} style={{ strokeWidth: '1px' }} />
          }
          bgColor={'red.300'}
          color={'white'}
          height={'45px'}
          borderTopRadius={0}
          borderBottomRadius={12}
          _hover={{ bgColor: 'red.400' }}
          onClick={() => handleGameplayDeletion()}
        >
          Delete Gameplay
        </MenuItem>
      </MenuList>
    </Menu>
  );
};

export const getServerSideProps: GetServerSideProps = async ({ req, res }) => {
  const user = await getServerSession(req, res, authOptions);

  if (user?.user?.role === 'ADMIN' || user?.user?.role === 'MOD')
    return { props: {} };
  else return { redirect: { destination: '/404' }, props: {} };
};

Gameplay.getLayout = function getLayout(page: ReactElement) {
  return <Layout>{page}</Layout>;
};
